import os, time
import javalang
import ast
from neo4j import GraphDatabase
from pathlib import Path

# ENV
uri = "bolt://localhost:7687"
user = "neo4j"
password = "xxx"

class CodeKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_node(self, label, name, properties=None):
        with self.driver.session() as session:
            session.execute_write(self._add_node, label, name, properties)

    def add_relationship(self, from_label, from_name, to_label, to_name, relation, properties=None):
        with self.driver.session() as session:
            session.execute_write(self._add_relationship, from_label, from_name, to_label, to_name, relation, properties)

    def clear_database(self):
        print("Clearing database...")
        with self.driver.session() as session:
            session.execute_write(self._clear_database)

    @staticmethod
    def _add_node(tx, label, name, properties):
        props = ", ".join([f"{k}: '{v}'" for k, v in properties.items()]) if properties else ""
        query = f"MERGE (n:{label} {{name: '{name}' {', ' + props if props else ''}}})"
        tx.run(query)

    @staticmethod
    def _add_relationship(tx, from_label, from_name, to_label, to_name, relation, properties):
        props = ", ".join([f"{k}: '{v}'" for k, v in properties.items()]) if properties else ""
        query = f"""
        MATCH (a:{from_label} {{name: '{from_name}'}})
        MATCH (b:{to_label} {{name: '{to_name}'}})
        MERGE (a)-[r:{relation} {f'{{ {props} }}' if props else ''}]->(b)
        """
        tx.run(query)

    @staticmethod
    def _clear_database(tx):
        query = "MATCH (n) DETACH DELETE n"
        tx.run(query)

# 解析Java代码，提取关系构造知识图谱
def build_code_knowledge_graph(graph, code, project_name):
    tree = javalang.parse.parse(code)

    def process_class(graph, node, project_name, parent_class=None):
        graph.add_node(f'{project_name}Class', node.name)
        # 类关系：嵌套、继承、实现接口
        if parent_class:
            graph.add_relationship(f'{project_name}Class', parent_class, f'{project_name}Class', node.name, 'NESTED_IN')

        if node.extends:
            graph.add_relationship(f'{project_name}Class', node.name, f'{project_name}Class', node.extends.name, 'INHERITS')

        if node.implements:
            for interface in node.implements:
                graph.add_relationship(f'{project_name}Class', node.name, f'{project_name}Interface', interface.name, 'IMPLEMENTS')

        # 类、方法、字段定义
        for member in node.body:
            if isinstance(member, javalang.tree.ClassDeclaration):
                process_class(graph, member, project_name= project_name, parent_class=node.name)
            elif isinstance(member, javalang.tree.MethodDeclaration):
                process_method(graph, node, member, project_name=project_name)
            # elif isinstance(member, javalang.tree.FieldDeclaration):
            #     process_field(graph, node, member, project_name=project_name)

    def process_method(graph, class_node, method_node, project_name):
        graph.add_node(f'{project_name}Method', method_node.name)
        graph.add_relationship(f'{project_name}Class', class_node.name, f'{project_name}Method', method_node.name, 'HAS_METHOD')

        return_type = method_node.return_type.name if method_node.return_type else 'void'
        graph.add_relationship(f'{project_name}Method', method_node.name, f'{project_name}Type', return_type, 'RETURNS')

        # for parameter in method_node.parameters:
        #     graph.add_node(f'{project_name}Parameter', parameter.name)
        #     graph.add_relationship(f'{project_name}Method', method_node.name, f'{project_name}Parameter', parameter.name, 'HAS_PARAMETER')

        for path, call in method_node.filter(javalang.tree.MethodInvocation):
            graph.add_node(f'{project_name}Method', call.member)
            graph.add_relationship(f'{project_name}Method', method_node.name, f'{project_name}Method', call.member, 'CALLS')

    # def process_field(graph, class_node, field_node, project_name):
    #     for declarator in field_node.declarators:
    #         graph.add_node(f'{project_name}Field', declarator.name, {'type': field_node.type.name})
    #         graph.add_relationship(f'{project_name}Class', class_node.name, f'{project_name}Field', declarator.name, 'HAS_FIELD')

    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        process_class(graph, node, project_name=project_name)


def build_python_knowledge_graph(graph, code, project_name):
    tree = ast.parse(code)

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self, graph, project_name):
            self.graph = graph
            self.project_name = project_name
            self.current_class = None
            self.current_function = None

        def visit_ClassDef(self, node):
            class_name = node.name
            self.graph.add_node(f'{self.project_name}Class', class_name)

            # 继承关系
            for base in node.bases:
                if isinstance(base, ast.Name):
                    self.graph.add_relationship(f'{self.project_name}Class', class_name, f'{self.project_name}Class', base.id, 'INHERITS')

            prev_class = self.current_class
            self.current_class = class_name
            self.generic_visit(node)
            self.current_class = prev_class

        def visit_FunctionDef(self, node):
            func_name = node.name
            self.graph.add_node(f'{self.project_name}Method', func_name)

            # 所属类
            if self.current_class:
                self.graph.add_relationship(f'{self.project_name}Class', self.current_class, f'{self.project_name}Method', func_name, 'HAS_METHOD')

            # # 参数类型
            # for arg in node.args.args:
            #     self.graph.add_node(f'{self.project_name}Parameter', arg.arg)
            #     self.graph.add_relationship(f'{self.project_name}Method', func_name, f'{self.project_name}Parameter', arg.arg, 'HAS_PARAMETER')

            # 收集函数调用
            prev_func = self.current_function
            self.current_function = func_name
            self.generic_visit(node)
            self.current_function = prev_func

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            else:
                callee = None

            if callee and self.current_function:
                self.graph.add_node(f'{self.project_name}Method', callee)
                self.graph.add_relationship(f'{self.project_name}Method', self.current_function, f'{self.project_name}Method', callee, 'CALLS')

            self.generic_visit(node)

        # def visit_Assign(self, node):
        #     if self.current_class:
        #         for target in node.targets:
        #             if isinstance(target, ast.Name):
        #                 var_name = target.id
        #                 self.graph.add_node(f'{self.project_name}Field', var_name)
        #                 self.graph.add_relationship(f'{self.project_name}Class', self.current_class, f'{self.project_name}Field', var_name, 'HAS_FIELD')

    visitor = FunctionVisitor(graph, project_name)
    visitor.visit(tree)


# 构建知识图谱
def build_project_knowledge_graph(graph, project_path, project_name):
    print(f"Building knowledge graph for project {project_name}...")
    time1 = time.time()
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".java"):
                file_path = os.path.join(root, file)
                process_java_file(graph, file_path, project_name)
    time2 = time.time()
    print(f"Knowledge graph for project {project_name} built in {time2 - time1:.2f} seconds")

def process_java_file(graph, file_path, project_name):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            code = f.read()
            build_code_knowledge_graph(graph, code, project_name)
        except javalang.parser.JavaSyntaxError:
            print(f"Syntax error in file {file_path}, skipping...")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def build_project_knowledge_graph_python(graph, project_path, project_name):
    print(f"Building knowledge graph for Python project {project_name}...")
    time1 = time.time()
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                process_python_file(graph, file_path, project_name)
    time2 = time.time()
    print(f"Knowledge graph for Python project {project_name} built in {time2 - time1:.2f} seconds")

def process_python_file(graph, file_path, project_name):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[WARN] File not found: {file_path}")
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            build_python_knowledge_graph(graph, code, project_name)
    except SyntaxError as e:
        print(f"[WARN] Syntax error in file {file_path}: {e}, skipping...")
    except Exception as e:
        print(f"[ERROR] Error processing file {file_path}: {e}")


if __name__ == '__main__':
    graph = CodeKnowledgeGraph(uri, user, password)
    graph.clear_database()

    #Pytorch
    build_project_knowledge_graph_python(graph, '../data/code/pytorch', 'pytorch')

    # Tensorflow
    build_project_knowledge_graph_python(graph, '../data/code/tensorflow', 'tensorflow')

    #Keras
    build_project_knowledge_graph_python(graph, '../data/code/keras', 'keras')

    graph.close()