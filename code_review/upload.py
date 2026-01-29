"""
Git Diff Code Reviewer with Functional Duplication Detection
Reviews uncommitted changes on current branch
Detects functional duplicates, risks, and confidential data exposure
"""
import subprocess
import re
import ast
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate


@dataclass
class ReviewConfig:
    """Configuration for diff reviewer"""
    bot_path: str = "./rpa_bot"
    codebase_path: str = "./rpa_codebase"
    persist_dir: str = "./chroma_db"
    bot_collection_name: str = "bot"
    codebase_collection_name: str = "codebase"
    ollama_host: str = "http://localhost:11434"
    chat_model: str = "py-chat"
    embed_model: str = "py-embed"
    functional_similarity_threshold: float = 0.7  # 70% similar = duplicate


@dataclass
class ChangedFunction:
    """Represents a function that changed"""
    file_path: str
    function_name: str
    change_type: str  # "added", "modified", "deleted"
    old_code: Optional[str]
    new_code: Optional[str]
    line_start: int
    line_end: int


@dataclass
class FunctionalDuplicate:
    """Represents a functionally duplicate function"""
    function_name: str
    file_path: str
    source: str  # "bot" or "codebase"
    similarity_score: float
    reasons: List[str]
    code_snippet: str


class RiskDetector:
    """Detect security and reliability risks"""

    RISK_PATTERNS = {
        'sql_injection': [
            (r'execute\s*\(\s*["\'].*%s.*["\']', 'String formatting in SQL query'),
            (r'execute\s*\(\s*f["\']', 'F-string in SQL query'),
            (r'\.format\s*\(.*\).*execute', 'Format in SQL query'),
        ],
        'command_injection': [
            (r'os\.system\s*\(', 'os.system() call'),
            (r'subprocess\.(call|run|Popen).*shell=True', 'subprocess with shell=True'),
        ],
        'hardcoded_credentials': [
            (r'password\s*=\s*["\'][^"\']{3,}["\']', 'Hardcoded password'),
            (r'api_key\s*=\s*["\'][^"\']{10,}["\']', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']{3,}["\']', 'Hardcoded secret'),
        ],
        'resource_leak': [
            (r'(?<!with\s.{0,50})open\s*\([^)]+\)(?!\s+as\s)', 'File opened without context manager'),
        ],
        'dangerous_functions': [
            (r'\beval\s*\(', 'eval() usage'),
            (r'\bexec\s*\(', 'exec() usage'),
        ],
        'error_handling': [
            (r'except\s*:\s*(?!.*raise)', 'Bare except without re-raise'),
            (r'except\s+Exception\s*:\s*pass', 'Exception swallowed silently'),
        ]
    }

    @classmethod
    def detect(cls, code: str) -> Dict[str, List[Dict]]:
        """Detect all risks in code"""
        findings = {}

        for risk_type, patterns in cls.RISK_PATTERNS.items():
            matches = []
            for pattern, description in patterns:
                for match in re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE):
                    line_num = code[:match.start()].count('\n') + 1
                    matches.append({
                        'line': line_num,
                        'description': description,
                        'code': match.group(0)[:50]
                    })

            if matches:
                findings[risk_type] = matches

        return findings


class ConfidentialDataScanner:
    """Scan for exposed confidential data"""

    PATTERNS = {
        'password': [
            (r'password\s*=\s*["\']([^"\']{3,})["\']', 'Hardcoded password'),
        ],
        'api_key': [
            (r'api[_-]?key\s*=\s*["\']([^"\']{10,})["\']', 'API key'),
        ],
        'database_url': [
            (r'(postgresql|mysql)://[^"\'\s]+', 'Database connection string'),
        ],
        'email': [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email address'),
        ],
        'ip_address': [
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP address'),
        ]
    }

    SAFE_PATTERNS = [
        r'os\.environ', r'getenv', r'config\.',
        r'example\.com', r'localhost', r'127\.0\.0\.1',
        r'YOUR_.*_HERE', r'<.*>', r'\.\.\.', r'xxx'
    ]

    @classmethod
    def scan(cls, code: str) -> Dict[str, List[Dict]]:
        """Scan for confidential data"""
        findings = {}

        for data_type, patterns in cls.PATTERNS.items():
            matches = []
            for pattern, description in patterns:
                for match in re.finditer(pattern, code, re.IGNORECASE):
                    # Check if this is a safe pattern (env var, config, placeholder)
                    context = code[max(0, match.start()-100):match.end()+50]
                    if any(re.search(safe, context) for safe in cls.SAFE_PATTERNS):
                        continue

                    value = match.group(1) if match.groups() else match.group(0)
                    line_num = code[:match.start()].count('\n') + 1

                    matches.append({
                        'line': line_num,
                        'description': description,
                        'value_preview': value[:20] + '***'
                    })

            if matches:
                findings[data_type] = matches

        return findings


class FunctionalAnalyzer:
    """Analyze what a function does (not how it does it)"""

    # Map of operations to keywords
    OPERATION_KEYWORDS = {
        'excel_read': ['read_excel', 'load_workbook', 'openpyxl', 'xlrd', 'ExcelFile'],
        'excel_write': ['to_excel', 'save_workbook', 'ExcelWriter'],
        'csv_read': ['read_csv', 'csv.reader', 'DictReader'],
        'csv_write': ['to_csv', 'csv.writer', 'DictWriter'],
        'database_read': ['select', 'query', 'fetchall', 'fetchone'],
        'database_write': ['insert', 'update', 'delete', 'commit'],
        'database_transaction': ['begin', 'commit', 'rollback', 'transaction'],
        'api_call': ['requests.get', 'requests.post', 'http.client', 'urllib'],
        'web_scraping': ['BeautifulSoup', 'soup.find', 'selenium', 'webdriver'],
        'web_automation': ['selenium', 'webdriver', 'click()', 'send_keys'],
        'authentication': ['login', 'authenticate', 'auth', 'credentials'],
        'validation': ['validate', 'check', 'verify', 'assert', 'isinstance'],
        'data_transformation': ['transform', 'convert', 'parse', 'normalize'],
        'file_read': ['open(', 'read()', 'readlines'],
        'file_write': ['write()', 'writelines', 'dump'],
        'email': ['smtplib', 'send_mail', 'EmailMessage'],
        'pdf': ['pypdf', 'reportlab', 'pdfplumber'],
        'logging': ['logger.', 'logging.', 'log.'],
        'error_handling': ['try:', 'except', 'raise', 'finally'],
    }

    @staticmethod
    def extract_operations(code: str) -> Set[str]:
        """Extract what operations this code performs"""
        operations = set()
        code_lower = code.lower()

        for operation, keywords in FunctionalAnalyzer.OPERATION_KEYWORDS.items():
            if any(kw.lower() in code_lower for kw in keywords):
                operations.add(operation)

        return operations

    @staticmethod
    def extract_imports(code: str) -> Set[str]:
        """Extract imported libraries"""
        imports = set()

        # Match: import pandas, from pandas import ...
        for match in re.finditer(r'(?:^|\n)\s*(?:import|from)\s+(\w+)', code):
            imports.add(match.group(1))

        return imports

    @staticmethod
    def extract_function_name_keywords(func_name: str) -> Set[str]:
        """Extract meaningful keywords from function name"""
        # Split by underscore and camelCase
        parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', func_name)
        parts.extend(func_name.split('_'))

        # Filter out common words
        common = {'get', 'set', 'do', 'make', 'run', 'execute', 'process', 'handle'}
        keywords = {p.lower() for p in parts if p.lower() not in common and len(p) > 2}

        return keywords


class FunctionalDuplicationDetector:
    """Detect when different code does the same thing"""

    def __init__(self, llm, embeddings, bot_vectorstore, codebase_vectorstore,
                 bot_path, codebase_path, threshold=0.7):
        self.llm = llm
        self.embeddings = embeddings
        self.bot_vectorstore = bot_vectorstore
        self.codebase_vectorstore = codebase_vectorstore
        self.bot_path = Path(bot_path)
        self.codebase_path = Path(codebase_path)
        self.threshold = threshold

        # Cache all functions from bot (for faster comparison)
        self.bot_functions_cache = self._load_all_bot_functions()

    def _load_all_bot_functions(self) -> List[Dict]:
        """Load all functions from bot filesystem"""
        all_functions = []

        print("📦 Loading bot functions for comparison...", end=" ")

        for py_file in self.bot_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                functions = self._parse_functions(content)
                for func in functions:
                    func['file'] = str(py_file.relative_to(self.bot_path))
                    func['operations'] = FunctionalAnalyzer.extract_operations(func['code'])
                    func['imports'] = FunctionalAnalyzer.extract_imports(func['code'])
                    all_functions.append(func)
            except:
                continue

        print(f"✅ {len(all_functions)} functions loaded")
        return all_functions

    def _parse_functions(self, code: str) -> List[Dict]:
        """Parse functions from code using AST"""
        try:
            tree = ast.parse(code)
        except:
            return []

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_lines)

                functions.append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'code': func_code
                })

        return functions

    def find_duplicates(self, new_code: str, new_name: str) -> Dict[str, List[FunctionalDuplicate]]:
        """Find functional duplicates in both bot and codebase"""

        # Extract characteristics of new function
        new_operations = FunctionalAnalyzer.extract_operations(new_code)
        new_imports = FunctionalAnalyzer.extract_imports(new_code)
        new_name_keywords = FunctionalAnalyzer.extract_function_name_keywords(new_name)
        new_purpose = self._get_function_purpose(new_code)

        # Find duplicates in bot
        bot_duplicates = self._find_bot_duplicates(
            new_code, new_name, new_operations, new_imports,
            new_name_keywords, new_purpose
        )

        # Find duplicates in codebase
        codebase_duplicates = self._find_codebase_duplicates(
            new_code, new_operations, new_imports,
            new_name_keywords, new_purpose
        )

        return {
            'bot': bot_duplicates,
            'codebase': codebase_duplicates
        }

    def _find_bot_duplicates(self, new_code, new_name, new_ops, new_imports,
                            new_keywords, new_purpose) -> List[FunctionalDuplicate]:
        """Find duplicates in bot code"""
        duplicates = []

        for func in self.bot_functions_cache:
            # Skip self
            if func['name'] == new_name:
                continue

            # Calculate similarity
            score = self._calculate_similarity(
                new_ops, new_imports, new_keywords, new_purpose,
                func['operations'], func['imports'],
                func['name'], func['code']
            )

            if score >= self.threshold:
                reasons = self._explain_similarity(
                    new_ops, new_imports, new_keywords,
                    func['operations'], func['imports'], func['name']
                )

                duplicates.append(FunctionalDuplicate(
                    function_name=func['name'],
                    file_path=func['file'],
                    source='bot',
                    similarity_score=score,
                    reasons=reasons,
                    code_snippet=func['code'][:200]
                ))

        return sorted(duplicates, key=lambda x: x.similarity_score, reverse=True)

    def _find_codebase_duplicates(self, new_code, new_ops, new_imports,
                                  new_keywords, new_purpose) -> List[FunctionalDuplicate]:
        """Find duplicates in codebase using vectorstore"""
        duplicates = []

        # Strategy: Use vector search with full code
        candidates = self.codebase_vectorstore.similarity_search(
            query=new_code,  # FULL CODE, not just 500 chars
            k=30  # Get more candidates
        )

        for candidate in candidates:
            cand_code = candidate.page_content
            cand_ops = FunctionalAnalyzer.extract_operations(cand_code)
            cand_imports = FunctionalAnalyzer.extract_imports(cand_code)
            cand_name = candidate.metadata.get('element_name', 'unknown')

            score = self._calculate_similarity(
                new_ops, new_imports, new_keywords, new_purpose,
                cand_ops, cand_imports, cand_name, cand_code
            )

            if score >= self.threshold:
                reasons = self._explain_similarity(
                    new_ops, new_imports, new_keywords,
                    cand_ops, cand_imports, cand_name
                )

                duplicates.append(FunctionalDuplicate(
                    function_name=cand_name,
                    file_path=candidate.metadata.get('filename', 'unknown'),
                    source='codebase',
                    similarity_score=score,
                    reasons=reasons,
                    code_snippet=cand_code[:200]
                ))

        return sorted(duplicates, key=lambda x: x.similarity_score, reverse=True)[:5]

    def _calculate_similarity(self, ops1, imports1, keywords1, purpose1,
                             ops2, imports2, name2, code2) -> float:
        """Calculate functional similarity score (0.0 to 1.0)"""

        score = 0.0

        # 1. Operation overlap (40% weight) - MOST IMPORTANT
        if ops1 and ops2:
            ops_overlap = len(ops1 & ops2) / len(ops1 | ops2)
            score += ops_overlap * 0.4

        # 2. Import overlap (20% weight)
        if imports1 and imports2:
            import_overlap = len(imports1 & imports2) / len(imports1 | imports2)
            score += import_overlap * 0.2

        # 3. Function name keyword overlap (15% weight)
        keywords2 = FunctionalAnalyzer.extract_function_name_keywords(name2)
        if keywords1 and keywords2:
            name_overlap = len(keywords1 & keywords2) / len(keywords1 | keywords2)
            score += name_overlap * 0.15

        # 4. Purpose semantic similarity (25% weight)
        purpose2 = self._get_function_purpose(code2)
        purpose_sim = self._semantic_similarity(purpose1, purpose2)
        score += purpose_sim * 0.25

        return score

    def _get_function_purpose(self, code: str) -> str:
        """Get one-sentence purpose of function"""
        # Extract docstring if available
        docstring_match = re.search(r'"""(.+?)"""', code, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            # Take first line
            return docstring.split('\n')[0].strip()

        # Fallback: ask LLM (cached would be ideal)
        prompt = f"""What does this function do? Answer in ONE short sentence (max 10 words).

{code[:800]}

Purpose:"""

        try:
            purpose = self.llm.invoke(prompt).strip()
            return purpose[:100]  # Limit length
        except:
            return ""

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        if not text1 or not text2:
            return 0.0

        try:
            emb1 = np.array(self.embeddings.embed_query(text1)).reshape(1, -1)
            emb2 = np.array(self.embeddings.embed_query(text2)).reshape(1, -1)

            sim = cosine_similarity(emb1, emb2)[0][0]
            return float(sim)
        except:
            return 0.0

    def _explain_similarity(self, ops1, imports1, keywords1,
                           ops2, imports2, name2) -> List[str]:
        """Explain why functions are similar"""
        reasons = []

        # Operations overlap
        common_ops = ops1 & ops2
        if common_ops:
            reasons.append(f"Both perform: {', '.join(common_ops)}")

        # Imports overlap
        common_imports = imports1 & imports2
        if common_imports:
            reasons.append(f"Both use: {', '.join(common_imports)}")

        # Name keywords
        keywords2 = FunctionalAnalyzer.extract_function_name_keywords(name2)
        common_keywords = keywords1 & keywords2
        if common_keywords:
            reasons.append(f"Similar naming: {', '.join(common_keywords)}")

        return reasons


class GitDiffParser:
    """Parse git diff for working directory changes"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)

    def get_changed_files(self) -> List[str]:
        """Get changed Python files"""
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        files = result.stdout.strip().split('\n')
        return [f for f in files if f.endswith('.py') and f]

    def get_changed_line_ranges(self, file_path: str) -> List[Tuple[int, int]]:
        """Get changed line ranges"""
        result = subprocess.run(
            ["git", "diff", "--unified=0", "HEAD", "--", file_path],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        ranges = []
        for line in result.stdout.split('\n'):
            if line.startswith('@@'):
                match = re.search(r'\+(\d+)(?:,(\d+))?', line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    end = start + count - 1
                    ranges.append((start, end))

        return ranges

    def get_old_file_content(self, file_path: str) -> Optional[str]:
        """Get file from HEAD"""
        result = subprocess.run(
            ["git", "show", f"HEAD:{file_path}"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        return result.stdout if result.returncode == 0 else None


class CodeParser:
    """Parse Python code"""

    @staticmethod
    def parse_functions(code: str) -> List[Dict]:
        """Extract all functions"""
        try:
            tree = ast.parse(code)
        except:
            return []

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_lines)

                functions.append({
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'code': func_code
                })
            elif isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_lines = code.split('\n')[item.lineno - 1:item.end_lineno]
                        method_code = '\n'.join(method_lines)

                        functions.append({
                            'name': f"{node.name}.{item.name}",
                            'line_start': item.lineno,
                            'line_end': item.end_lineno,
                            'code': method_code
                        })

        return functions


class DiffReviewer:
    """Review uncommitted changes with functional duplication detection"""

    def __init__(self, config: ReviewConfig):
        self.config = config
        self.bot_path = Path(config.bot_path)
        self.codebase_path = Path(config.codebase_path)

        # Initialize LLM
        print(f"🤖 Initializing LLM: {config.chat_model}")
        self.llm = Ollama(
            model=config.chat_model,
            base_url=config.ollama_host
        )

        # Initialize embeddings
        print(f"🔧 Initializing embeddings: {config.embed_model}")
        self.embeddings = OllamaEmbeddings(
            model=config.embed_model,
            base_url=config.ollama_host
        )

        # Initialize VectorStores
        print(f"📚 Loading vectorstores from {config.persist_dir}")
        self.bot_vectorstore = Chroma(
            collection_name=config.bot_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_dir
        )

        self.codebase_vectorstore = Chroma(
            collection_name=config.codebase_collection_name,
            embedding_function=self.embeddings,
            persist_directory=config.persist_dir
        )

        # Initialize analyzers
        self.git_parser = GitDiffParser(config.bot_path)
        self.code_parser = CodeParser()

        # Initialize functional duplication detector
        self.dup_detector = FunctionalDuplicationDetector(
            llm=self.llm,
            embeddings=self.embeddings,
            bot_vectorstore=self.bot_vectorstore,
            codebase_vectorstore=self.codebase_vectorstore,
            bot_path=config.bot_path,
            codebase_path=config.codebase_path,
            threshold=config.functional_similarity_threshold
        )

    def identify_changed_functions(self) -> List[ChangedFunction]:
        """Identify changed functions"""
        changed_functions = []

        changed_files = self.git_parser.get_changed_files()

        if not changed_files:
            print("\n✅ No uncommitted changes found!")
            return []

        print(f"\n📝 Found {len(changed_files)} changed files:")
        for f in changed_files:
            print(f"  - {f}")

        for file_path in changed_files:
            print(f"\n🔍 Analyzing {file_path}...")

            changed_ranges = self.git_parser.get_changed_line_ranges(file_path)

            try:
                new_code = self._read_file(file_path)
            except FileNotFoundError:
                continue

            new_functions = self.code_parser.parse_functions(new_code)

            old_code = self.git_parser.get_old_file_content(file_path)
            old_functions = self.code_parser.parse_functions(old_code) if old_code else []
            old_funcs_dict = {f['name']: f for f in old_functions}

            affected_functions = set()
            for start, end in changed_ranges:
                for func in new_functions:
                    if not (end < func['line_start'] or start > func['line_end']):
                        affected_functions.add(func['name'])

            print(f"  📌 {len(affected_functions)} functions affected")

            for func_name in affected_functions:
                new_func = next((f for f in new_functions if f['name'] == func_name), None)
                if not new_func:
                    continue

                old_func = old_funcs_dict.get(func_name)

                if old_func is None:
                    change_type = "added"
                    old_code_str = None
                    print(f"    ✨ {func_name}: NEW")
                elif old_func['code'].strip() == new_func['code'].strip():
                    continue
                else:
                    change_type = "modified"
                    old_code_str = old_func['code']
                    print(f"    ✏️  {func_name}: MODIFIED")

                changed_functions.append(ChangedFunction(
                    file_path=file_path,
                    function_name=func_name,
                    change_type=change_type,
                    old_code=old_code_str,
                    new_code=new_func['code'],
                    line_start=new_func['line_start'],
                    line_end=new_func['line_end']
                ))

        return changed_functions

    def _read_file(self, file_path: str) -> str:
        """Read file from working directory"""
        full_path = self.bot_path / file_path
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()

    def review_changed_function(self, changed_func: ChangedFunction) -> Dict:
        """Review a single changed function"""
        print(f"  🔬 Reviewing {changed_func.function_name}...")

        # 1. Detect risks
        print(f"    🔍 Scanning for risks...", end=" ")
        risks = RiskDetector.detect(changed_func.new_code)
        print(f"{'⚠️ ' + str(len(risks)) + ' types' if risks else '✅'}")

        # 2. Scan confidential data
        print(f"    🔒 Scanning for confidential data...", end=" ")
        confidential = ConfidentialDataScanner.scan(changed_func.new_code)
        print(f"{'⚠️ ' + str(len(confidential)) + ' types' if confidential else '✅'}")

        # 3. Find functional duplicates
        print(f"    🔎 Detecting functional duplicates...", end=" ")
        duplicates = self.dup_detector.find_duplicates(
            changed_func.new_code,
            changed_func.function_name
        )
        total_dups = len(duplicates['bot']) + len(duplicates['codebase'])
        print(f"{'⚠️ ' + str(total_dups) + ' found' if total_dups else '✅'}")

        # 4. Find usage
        print(f"    📍 Finding usage...", end=" ")
        usage = self.bot_vectorstore.similarity_search(
            query=f"{changed_func.function_name} called invoke usage",
            k=5
        )
        print(f"{'✅ ' + str(len(usage)) + ' places' if usage else '⚠️ Not used'}")

        # 5. Generate LLM review
        print(f"    🤖 Generating review...", end=" ")
        llm_review = self._generate_llm_review(
            changed_func, risks, confidential, duplicates, usage
        )
        print("✅")

        return {
            "file": changed_func.file_path,
            "function": changed_func.function_name,
            "line_start": changed_func.line_start,
            "line_end": changed_func.line_end,
            "change_type": changed_func.change_type,
            "risks": risks,
            "confidential_data": confidential,
            "duplicates": duplicates,
            "usage": usage,
            "llm_review": llm_review
        }

    def _generate_llm_review(self, func, risks, confidential, duplicates, usage):
        """Generate LLM review with all context"""

        # Build automated findings
        automated_findings = []

        if risks:
            automated_findings.append("🔴 SECURITY/RELIABILITY RISKS DETECTED:")
            for risk_type, findings in risks.items():
                automated_findings.append(f"  - {risk_type}: {len(findings)} issue(s)")

        if confidential:
            automated_findings.append("🔴 CONFIDENTIAL DATA EXPOSURE:")
            for data_type, findings in confidential.items():
                automated_findings.append(f"  - {data_type}: {len(findings)} instance(s)")

        bot_dups = duplicates['bot']
        if bot_dups:
            automated_findings.append("🟡 FUNCTIONAL DUPLICATES IN BOT:")
            for dup in bot_dups[:3]:
                automated_findings.append(
                    f"  - {dup.function_name} in {dup.file_path} "
                    f"({dup.similarity_score*100:.0f}% similar)"
                )
                for reason in dup.reasons:
                    automated_findings.append(f"    • {reason}")

        codebase_dups = duplicates['codebase']
        if codebase_dups:
            automated_findings.append("🟡 BETTER IMPLEMENTATIONS IN CODEBASE:")
            for dup in codebase_dups[:3]:
                automated_findings.append(
                    f"  - {dup.function_name} in {dup.file_path} "
                    f"({dup.similarity_score*100:.0f}% similar)"
                )
                for reason in dup.reasons:
                    automated_findings.append(f"    • {reason}")

        if not usage:
            automated_findings.append("🟡 USAGE WARNING: Function may not be called anywhere")

        automated_summary = "\n".join(automated_findings) if automated_findings else "✅ No automated issues detected"

        # Build prompt
        if func.change_type == "added":
            prompt = f"""Review this NEW function:

FUNCTION: {func.function_name} (lines {func.line_start}-{func.line_end})
FILE: {func.file_path}

CODE:
{func.new_code}

AUTOMATED ANALYSIS:
{automated_summary}

USAGE:
{self._format_usage(usage)}

Provide a concise review focusing on:
1. Any issues not caught by automated analysis
2. Code quality and best practices
3. Recommendations

Keep it brief and actionable."""

        else:  # modified
            prompt = f"""Review this MODIFIED function:

FUNCTION: {func.function_name} (lines {func.line_start}-{func.line_end})
FILE: {func.file_path}

OLD CODE:
{func.old_code[:500] if func.old_code else 'N/A'}

NEW CODE:
{func.new_code}

AUTOMATED ANALYSIS:
{automated_summary}

Provide a concise review focusing on:
1. What changed and why it matters
2. Any issues not caught by automated analysis
3. Impact on callers
4. Recommendations

Keep it brief and actionable."""

        return self.llm.invoke(prompt)

    def _format_usage(self, usage) -> str:
        """Format usage results"""
        if not usage:
            return "⚠️ No usage found"

        snippets = []
        for doc in usage[:3]:
            snippet = doc.page_content[:150].replace('\n', ' ')
            snippets.append(f"- {snippet}...")

        return "\n".join(snippets)

    def review_current_changes(self) -> List[Dict]:
        """Review all uncommitted changes"""
        print("="*80)
        print("🔍 REVIEWING UNCOMMITTED CHANGES")
        print("="*80)

        changed_functions = self.identify_changed_functions()

        if not changed_functions:
            return []

        print(f"\n📊 Total functions to review: {len(changed_functions)}\n")

        reviews = []
        for func in changed_functions:
            review = self.review_changed_function(func)
            reviews.append(review)

        return reviews

    def generate_report(self, reviews: List[Dict]) -> str:
        """Generate markdown report"""
        if not reviews:
            return "✅ No changes to review!"

        lines = []
        lines.append("# 🤖 Code Review - Uncommitted Changes\n")

        # Statistics
        total_risks = sum(len(r['risks']) for r in reviews)
        total_confidential = sum(len(r['confidential_data']) for r in reviews)
        total_bot_dups = sum(len(r['duplicates']['bot']) for r in reviews)
        total_codebase_dups = sum(len(r['duplicates']['codebase']) for r in reviews)

        lines.append("## 📊 Summary\n")
        lines.append(f"- **Functions reviewed:** {len(reviews)}")
        lines.append(f"- **Security/reliability risks:** {total_risks}")
        lines.append(f"- **Confidential data exposures:** {total_confidential}")
        lines.append(f"- **Duplicates in bot:** {total_bot_dups}")
        lines.append(f"- **Better alternatives in codebase:** {total_codebase_dups}\n")

        # Group by file
        by_file = {}
        for review in reviews:
            file = review['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(review)

        # Per-file reviews
        for file_path, file_reviews in by_file.items():
            lines.append(f"## 📄 {file_path}\n")

            for rev in file_reviews:
                icon = "✨" if rev['change_type'] == "added" else "✏️"
                lines.append(f"### {icon} `{rev['function']}()` (lines {rev['line_start']}-{rev['line_end']})\n")
                lines.append(f"**Change Type:** {rev['change_type'].title()}\n")

                # Risks
                if rev['risks']:
                    lines.append("#### 🔴 Security/Reliability Risks\n")
                    for risk_type, findings in rev['risks'].items():
                        lines.append(f"**{risk_type.replace('_', ' ').title()}:**")
                        for finding in findings:
                            lines.append(f"- Line {finding['line']}: {finding['description']}")
                        lines.append("")

                # Confidential data
                if rev['confidential_data']:
                    lines.append("#### 🔒 Confidential Data Exposure\n")
                    for data_type, findings in rev['confidential_data'].items():
                        lines.append(f"**{data_type.replace('_', ' ').title()}:**")
                        for finding in findings:
                            lines.append(f"- Line {finding['line']}: {finding['description']}")
                        lines.append("")

                # Duplicates
                if rev['duplicates']['bot']:
                    lines.append("#### 🟡 Functional Duplicates in Bot\n")
                    for dup in rev['duplicates']['bot'][:3]:
                        lines.append(f"**{dup.function_name}** in `{dup.file_path}` ({dup.similarity_score*100:.0f}% similar)")
                        for reason in dup.reasons:
                            lines.append(f"  - {reason}")
                        lines.append("")

                if rev['duplicates']['codebase']:
                    lines.append("#### 💡 Better Implementations Available in Codebase\n")
                    for dup in rev['duplicates']['codebase'][:3]:
                        lines.append(f"**{dup.function_name}** in `{dup.file_path}` ({dup.similarity_score*100:.0f}% similar)")
                        for reason in dup.reasons:
                            lines.append(f"  - {reason}")
                        lines.append("")

                # LLM review
                lines.append("#### 🤖 AI Review\n")
                lines.append(rev['llm_review'])
                lines.append("\n---\n")

        return "\n".join(lines)


def main():
    """Review uncommitted changes"""
    config = ReviewConfig(
        bot_path="./rpa_bot",
        codebase_path="./rpa_codebase",
        persist_dir="./chroma_db",
        chat_model="py-chat",
        embed_model="py-embed",
        functional_similarity_threshold=0.7
    )

    reviewer = DiffReviewer(config)

    # Review
    reviews = reviewer.review_current_changes()

    # Generate report
    report = reviewer.generate_report(reviews)

    # Display
    print("\n" + "="*80)
    print("📋 REVIEW REPORT")
    print("="*80)
    print(report)

    # Save
    with open("diff_review.md", "w", encoding='utf-8') as f:
        f.write(report)

    print("\n✅ Review saved to diff_review.md")


if __name__ == "__main__":
    main()