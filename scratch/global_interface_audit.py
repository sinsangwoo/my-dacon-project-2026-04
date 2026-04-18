import ast
import os
import glob
from collections import defaultdict

def get_py_files(root_dir):
    return glob.glob(os.path.join(root_dir, "**/*.py"), recursive=True)

class InterfaceAuditor(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.definitions = {} # name -> {args, defaults, is_method}
        self.calls = [] # {name, lineno, args, keywords}

    def visit_FunctionDef(self, node):
        arg_names = [arg.arg for arg in node.args.args]
        defaults_count = len(node.args.defaults)
        
        # Identify defaults
        defaults = {}
        for i, val in enumerate(reversed(node.args.defaults)):
            arg_name = arg_names[-(i+1)]
            defaults[arg_name] = True # We just care if it HAS a default
            
        is_method = False
        if arg_names and arg_names[0] == 'self':
            is_method = True
            arg_names = arg_names[1:]

        self.definitions[node.name] = {
            'args': arg_names,
            'defaults': defaults,
            'is_method': is_method,
            'node': node
        }
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            name = None
            
        if name:
            self.calls.append({
                'name': name,
                'lineno': node.lineno,
                'args_count': len(node.args),
                'keywords': [k.arg for k in node.keywords]
            })
        self.generic_visit(node)

def run_audit(root_dir):
    files = get_py_files(root_dir)
    all_definitions = {} # (file, name) -> info
    all_calls = [] # (file, call_info)
    
    # Global map for definitions to handle imports
    global_defs = defaultdict(list)

    for f in files:
        if "venv" in f or "scratch" in f or "audit" in f: continue
        with open(f, "r", encoding="utf-8") as source:
            try:
                tree = ast.parse(source.read())
                auditor = InterfaceAuditor(f)
                auditor.visit(tree)
                for name, info in auditor.definitions.items():
                    all_definitions[(f, name)] = info
                    global_defs[name].append((f, info))
                for call in auditor.calls:
                    all_calls.append((f, call))
            except Exception as e:
                print(f"Error parsing {f}: {e}")

    mismatches = []
    
    for caller_file, call in all_calls:
        name = call['name']
        if name not in global_defs: continue # Internal or external lib
        
        # Try to find the matching definition. This is simplified (ignores imports).
        # We look for all definitions with this name and check for mismatches.
        possible_defs = global_defs[name]
        
        for def_file, def_info in possible_defs:
            # Skip checking if it's likely a standard lib or external call with same name
            # For this competition, we focus on src and main.
            if "src" not in def_file and "main.py" not in def_file: continue
            
            # Validation logic
            required_args = [a for a in def_info['args'] if a not in def_info['defaults']]
            max_args = len(def_info['args'])
            min_args = len(required_args)
            
            # 1. Positional check
            if call['args_count'] > max_args:
                mismatches.append({
                    'file': caller_file,
                    'line': call['lineno'],
                    'func': name,
                    'type': 'TOO_MANY_POSITIONAL_ARGS',
                    'detail': f"Call has {call['args_count']}, Def has max {max_args} in {def_file}"
                })
            
            # 2. Required check (Simplified, doesn't account for positional filling required args)
            total_provided = call['args_count'] + len(call['keywords'])
            if total_provided < min_args:
                 mismatches.append({
                    'file': caller_file,
                    'line': call['lineno'],
                    'func': name,
                    'type': 'TOO_FEW_ARGS',
                    'detail': f"Call has {total_provided}, Def requires min {min_args} in {def_file}"
                })
            
            # 3. Keyword mismatch
            for kw in call['keywords']:
                if kw not in def_info['args'] and kw is not None:
                    # Could be **kwargs, but let's check def node for that
                    has_kwargs = def_info['node'].args.kwarg is not None
                    if not has_kwargs:
                        mismatches.append({
                            'file': caller_file,
                            'line': call['lineno'],
                            'func': name,
                            'type': 'UNEXPECTED_KEYWORD',
                            'detail': f"Keyword '{kw}' not in signature of {def_file}"
                        })

    return mismatches

if __name__ == "__main__":
    results = run_audit(".")
    print("--- [GLOBAL INTERFACE MISMATCH LIST] ---")
    if not results:
        print("No mismatches found in project-internal calls.")
    else:
        for m in sorted(results, key=lambda x: (x['file'], x['line'])):
            print(f"FILE: {m['file']} | LINE: {m['line']} | FUNC: {m['func']} | TYPE: {m['type']}")
            print(f"  DETAIL: {m['detail']}")
    print("--- [/GLOBAL INTERFACE MISMATCH LIST] ---")
