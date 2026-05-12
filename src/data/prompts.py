"""Benchmark prompt dataset for speculative decoding evaluation.

Loads 150 prompts across three domains (code, conversation, summarization)
from standard HuggingFace datasets, falling back to hand-written examples
if a dataset is unavailable.  All prompts are pre-tokenized and stored with
their token count so downstream benchmarks can filter or stratify by length.

CLI usage::

    python -m src.data.prompts --tokenizer gpt2 --output data/prompts.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Domain = Literal["code", "conversation", "summarization"]

_DOMAINS: tuple[Domain, ...] = ("code", "conversation", "summarization")


# ---------------------------------------------------------------------------
# Fallback prompt banks
# Used when the corresponding HuggingFace dataset cannot be reached.
# ---------------------------------------------------------------------------

_FALLBACK_CODE: tuple[str, ...] = (
    # Math / number theory
    'def fibonacci(n: int) -> int:\n    """Return the nth Fibonacci number (0-indexed, fibonacci(0)=0, fibonacci(1)=1).\n\n    Args:\n        n: Non-negative integer index.\n\n    Returns:\n        The nth Fibonacci number.\n    """\n',
    'def factorial(n: int) -> int:\n    """Compute n! iteratively.\n\n    Args:\n        n: Non-negative integer.\n\n    Returns:\n        n factorial.\n    """\n',
    'def gcd(a: int, b: int) -> int:\n    """Return the greatest common divisor of a and b using the Euclidean algorithm.\n\n    Args:\n        a: First integer.\n        b: Second integer.\n\n    Returns:\n        GCD(a, b).\n    """\n',
    'def lcm(a: int, b: int) -> int:\n    """Return the least common multiple of a and b.\n\n    Args:\n        a: First integer.\n        b: Second integer.\n\n    Returns:\n        LCM(a, b).\n    """\n',
    'def is_prime(n: int) -> bool:\n    """Determine whether n is a prime number.\n\n    Args:\n        n: Integer to test.\n\n    Returns:\n        True if n is prime, False otherwise.\n    """\n',
    'def sieve_of_eratosthenes(limit: int) -> list[int]:\n    """Return all prime numbers up to limit using the Sieve of Eratosthenes.\n\n    Args:\n        limit: Upper bound (inclusive) for prime generation.\n\n    Returns:\n        Sorted list of primes <= limit.\n    """\n',
    'def integer_sqrt(n: int) -> int:\n    """Return the floor of the square root of n without floating-point arithmetic.\n\n    Args:\n        n: Non-negative integer.\n\n    Returns:\n        Largest integer k such that k*k <= n.\n    """\n',
    'def fast_power(base: int, exp: int, mod: int) -> int:\n    """Compute base**exp % mod using binary exponentiation.\n\n    Args:\n        base: The base value.\n        exp: Non-negative exponent.\n        mod: Positive modulus.\n\n    Returns:\n        (base ** exp) % mod.\n    """\n',
    'def combinations(n: int, k: int) -> int:\n    """Return the number of ways to choose k items from n without repetition.\n\n    Args:\n        n: Total number of items.\n        k: Number of items to choose.\n\n    Returns:\n        C(n, k).\n    """\n',
    'def permutations_count(n: int, k: int) -> int:\n    """Return the number of ordered arrangements of k items chosen from n.\n\n    Args:\n        n: Total number of items.\n        k: Number of items to arrange.\n\n    Returns:\n        P(n, k) = n! / (n-k)!.\n    """\n',
    # Strings
    'def reverse_string(s: str) -> str:\n    """Return s with characters in reverse order.\n\n    Args:\n        s: Input string.\n\n    Returns:\n        Reversed string.\n    """\n',
    'def is_palindrome(s: str) -> bool:\n    """Check whether s reads the same forwards and backwards (case-insensitive).\n\n    Args:\n        s: Input string.\n\n    Returns:\n        True if s is a palindrome.\n    """\n',
    'def count_vowels(s: str) -> int:\n    """Count the vowel characters (a, e, i, o, u) in s.\n\n    Args:\n        s: Input string (may contain uppercase letters).\n\n    Returns:\n        Number of vowels.\n    """\n',
    'def to_title_case(s: str) -> str:\n    """Convert s to title case without using str.title().\n\n    Args:\n        s: Input string.\n\n    Returns:\n        Title-cased string.\n    """\n',
    'def run_length_encode(s: str) -> str:\n    """Compress s using run-length encoding.\n\n    Example: "aaabbc" -> "3a2b1c".\n\n    Args:\n        s: Non-empty string of letters.\n\n    Returns:\n        Run-length encoded representation.\n    """\n',
    'def longest_common_prefix(words: list[str]) -> str:\n    """Find the longest string that is a prefix of every word in words.\n\n    Args:\n        words: Non-empty list of strings.\n\n    Returns:\n        Longest common prefix, or "" if none exists.\n    """\n',
    'def word_frequency(text: str) -> dict[str, int]:\n    """Count the frequency of each word in text (case-insensitive).\n\n    Args:\n        text: Whitespace-delimited input text.\n\n    Returns:\n        Mapping from lowercase word to its occurrence count.\n    """\n',
    'def is_anagram(a: str, b: str) -> bool:\n    """Determine whether a and b are anagrams of each other.\n\n    Args:\n        a: First string.\n        b: Second string.\n\n    Returns:\n        True if a and b contain the same characters in any order.\n    """\n',
    'def caesar_cipher(text: str, shift: int) -> str:\n    """Encrypt text with a Caesar cipher.\n\n    Args:\n        text: Plaintext string.\n        shift: Letter shift amount (can be negative for decryption).\n\n    Returns:\n        Encrypted string; non-letter characters are unchanged.\n    """\n',
    'def remove_duplicate_chars(s: str) -> str:\n    """Remove duplicate characters from s, keeping the first occurrence of each.\n\n    Args:\n        s: Input string.\n\n    Returns:\n        String with duplicate characters removed.\n    """\n',
    # Lists / arrays
    'def find_maximum(nums: list[int]) -> int:\n    """Return the largest integer in nums without using max().\n\n    Args:\n        nums: Non-empty list of integers.\n\n    Returns:\n        Maximum element.\n    """\n',
    'def binary_search(arr: list[int], target: int) -> int:\n    """Search for target in the sorted array arr.\n\n    Args:\n        arr: Sorted list of integers.\n        target: Value to find.\n\n    Returns:\n        Index of target, or -1 if not found.\n    """\n',
    'def merge_sorted_arrays(a: list[int], b: list[int]) -> list[int]:\n    """Merge two sorted arrays into a single sorted array.\n\n    Args:\n        a: Sorted list of integers.\n        b: Sorted list of integers.\n\n    Returns:\n        Merged sorted list.\n    """\n',
    'def flatten(nested: list) -> list:\n    """Recursively flatten an arbitrarily nested list.\n\n    Args:\n        nested: A list that may contain lists as elements at any depth.\n\n    Returns:\n        Flat list with all elements in depth-first order.\n    """\n',
    'def chunk(lst: list, size: int) -> list[list]:\n    """Split lst into consecutive chunks of at most size elements.\n\n    Args:\n        lst: Input list.\n        size: Maximum chunk length (positive integer).\n\n    Returns:\n        List of sublists.\n    """\n',
    'def rotate_left(arr: list, k: int) -> list:\n    """Rotate arr to the left by k positions.\n\n    Args:\n        arr: Input list.\n        k: Number of positions to rotate.\n\n    Returns:\n        Rotated list (original is not modified).\n    """\n',
    'def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:\n    """Find two indices in nums whose values sum to target.\n\n    Args:\n        nums: List of integers.\n        target: Target sum.\n\n    Returns:\n        Tuple (i, j) with i < j such that nums[i] + nums[j] == target,\n        or None if no such pair exists.\n    """\n',
    'def max_subarray_sum(nums: list[int]) -> int:\n    """Return the maximum sum of any contiguous subarray (Kadane\'s algorithm).\n\n    Args:\n        nums: List of integers (may be all negative).\n\n    Returns:\n        Maximum subarray sum.\n    """\n',
    'def running_average(nums: list[float]) -> list[float]:\n    """Compute the cumulative running average of nums.\n\n    Args:\n        nums: Non-empty list of numbers.\n\n    Returns:\n        List where element i is the mean of nums[0..i].\n    """\n',
    'def deduplicate_ordered(lst: list) -> list:\n    """Remove duplicate elements while preserving insertion order.\n\n    Args:\n        lst: Input list with hashable elements.\n\n    Returns:\n        List with duplicates removed, order preserved.\n    """\n',
    # Trees / graphs
    'def tree_height(root) -> int:\n    """Compute the height of a binary tree.\n\n    Args:\n        root: Root node with optional .left and .right attributes, or None.\n\n    Returns:\n        Height of the tree (0 for a single node, -1 for an empty tree).\n    """\n',
    'def level_order_traversal(root) -> list[list[int]]:\n    """Return the values of a binary tree grouped by level.\n\n    Args:\n        root: Root TreeNode (with .val, .left, .right), or None.\n\n    Returns:\n        List of levels; each level is a list of node values left-to-right.\n    """\n',
    'def is_valid_bst(root) -> bool:\n    """Determine whether a binary tree satisfies the BST property.\n\n    Args:\n        root: Root TreeNode with .val, .left, .right attributes.\n\n    Returns:\n        True if the tree is a valid binary search tree.\n    """\n',
    'def depth_first_search(graph: dict[int, list[int]], start: int) -> list[int]:\n    """Iterative depth-first search starting from start.\n\n    Args:\n        graph: Adjacency list mapping node -> list of neighbours.\n        start: Starting node.\n\n    Returns:\n        Nodes in DFS visitation order.\n    """\n',
    'def breadth_first_search(graph: dict[int, list[int]], start: int) -> list[int]:\n    """Breadth-first search starting from start.\n\n    Args:\n        graph: Adjacency list mapping node -> list of neighbours.\n        start: Starting node.\n\n    Returns:\n        Nodes in BFS visitation order.\n    """\n',
    'def has_cycle_directed(graph: dict[int, list[int]]) -> bool:\n    """Detect whether a directed graph contains a cycle.\n\n    Args:\n        graph: Adjacency list for a directed graph.\n\n    Returns:\n        True if at least one cycle exists.\n    """\n',
    'def count_islands(grid: list[list[int]]) -> int:\n    """Count connected groups of 1s (islands) in a binary grid.\n\n    Args:\n        grid: 2D list where 1 = land and 0 = water.\n\n    Returns:\n        Number of distinct islands.\n    """\n',
    'def topological_sort(graph: dict[int, list[int]]) -> list[int]:\n    """Return a topological ordering of a directed acyclic graph.\n\n    Args:\n        graph: Adjacency list for a DAG.\n\n    Returns:\n        Nodes in topological order.\n\n    Raises:\n        ValueError: If the graph contains a cycle.\n    """\n',
    'def dijkstra(graph: dict[str, dict[str, int]], src: str, dst: str) -> int:\n    """Shortest path from src to dst using Dijkstra\'s algorithm.\n\n    Args:\n        graph: Weighted adjacency dict {node: {neighbour: weight}}.\n        src: Source node.\n        dst: Destination node.\n\n    Returns:\n        Minimum total weight, or -1 if dst is unreachable.\n    """\n',
    'def is_connected(graph: dict[int, list[int]]) -> bool:\n    """Check whether an undirected graph is fully connected.\n\n    Args:\n        graph: Adjacency list for an undirected graph.\n\n    Returns:\n        True if every node is reachable from every other node.\n    """\n',
    # Utilities
    'def memoize(func):\n    """Return a memoised wrapper that caches results keyed by arguments.\n\n    Args:\n        func: Pure function with hashable arguments.\n\n    Returns:\n        Wrapped function that caches return values.\n    """\n',
    'def retry(max_attempts: int, exceptions: tuple = (Exception,)):\n    """Decorator factory that retries a function on specified exceptions.\n\n    Args:\n        max_attempts: Maximum number of calls before re-raising.\n        exceptions: Exception types that trigger a retry.\n\n    Returns:\n        Decorator wrapping the target function with retry logic.\n    """\n',
    'def parse_csv_line(line: str, delimiter: str = ",") -> list[str]:\n    """Parse a single CSV line, respecting quoted fields.\n\n    Args:\n        line: Raw CSV line string.\n        delimiter: Field separator character.\n\n    Returns:\n        List of field values with surrounding quotes stripped.\n    """\n',
    'def deep_merge(base: dict, override: dict) -> dict:\n    """Recursively merge override into base, with override taking priority.\n\n    Args:\n        base: Base dictionary.\n        override: Dictionary whose values win on key conflicts.\n\n    Returns:\n        New merged dictionary (inputs are not modified).\n    """\n',
    'def validate_email(address: str) -> bool:\n    """Check whether address is a syntactically valid email address.\n\n    Args:\n        address: Email address string to validate.\n\n    Returns:\n        True if address matches local@domain.tld structure.\n    """\n',
    'def format_bytes(n: int) -> str:\n    """Format a byte count as a human-readable string.\n\n    Examples: 1024 -> "1.0 KB", 1_048_576 -> "1.0 MB".\n\n    Args:\n        n: Non-negative byte count.\n\n    Returns:\n        Human-readable string with appropriate unit suffix.\n    """\n',
    'def clamp(value: float, lo: float, hi: float) -> float:\n    """Clamp value to the closed interval [lo, hi].\n\n    Args:\n        value: The value to clamp.\n        lo: Lower bound.\n        hi: Upper bound.\n\n    Returns:\n        lo if value < lo, hi if value > hi, else value.\n    """\n',
    'def moving_average(data: list[float], window: int) -> list[float]:\n    """Compute the simple moving average with a sliding window.\n\n    Args:\n        data: Sequence of numerical values.\n        window: Window size (positive integer).\n\n    Returns:\n        List of averages, using an expanding window for the first\n        window-1 positions.\n    """\n',
    'def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:\n    """Multiply two matrices a (m x k) and b (k x n).\n\n    Args:\n        a: m x k matrix as a list of rows.\n        b: k x n matrix as a list of rows.\n\n    Returns:\n        m x n product matrix.\n\n    Raises:\n        ValueError: If inner dimensions do not match.\n    """\n',
    'def transpose(matrix: list[list[float]]) -> list[list[float]]:\n    """Return the transpose of a 2D matrix.\n\n    Args:\n        matrix: m x n matrix as a list of rows.\n\n    Returns:\n        n x m transposed matrix.\n    """\n',
    'def levenshtein_distance(s: str, t: str) -> int:\n    """Compute the Levenshtein edit distance between strings s and t.\n\n    Args:\n        s: Source string.\n        t: Target string.\n\n    Returns:\n        Minimum number of single-character insertions, deletions, or\n        substitutions needed to transform s into t.\n    """\n',
)

_FALLBACK_CONVERSATION: tuple[str, ...] = (
    # Writing & editing
    "Write a professional email declining a job offer while expressing gratitude for the opportunity.",
    "Proofread and improve the following sentence for clarity and grammar: 'The team have decided to not proceed with the project because of various reasons that was identified.'",
    "Write a concise executive summary for a product launch announcement targeting enterprise customers.",
    "Translate the following into formal written English: 'Hey, just wanted to check if u got my message about the meeting tmrw?'",
    "Rewrite the following paragraph at a 6th-grade reading level: 'The mitochondria facilitates the synthesis of adenosine triphosphate through oxidative phosphorylation.'",
    "Write a cover letter for a software engineer with 3 years of experience applying to an AI infrastructure startup.",
    "Summarize the key steps of a typical software engineering interview process in bullet points.",
    "Draft a polite follow-up email to a client who has not responded to a proposal in two weeks.",
    "Write three alternative headlines for an article about the environmental impact of data centers.",
    "Convert the following bullet points into a single coherent paragraph: 'Python is readable. It has a large ecosystem. It is widely used in ML and data science.'",
    # Factual explanations
    "Explain how HTTPS works, including the role of TLS handshakes and certificate authorities.",
    "What is the difference between supervised, unsupervised, and reinforcement learning? Give one real-world example of each.",
    "Explain the CAP theorem and give a concrete example of a system that prioritises each trade-off.",
    "What caused the 2008 financial crisis? Explain the chain of events in plain language.",
    "How does GPS determine your precise location? Walk through the triangulation process.",
    "What is the difference between a virus and a bacterium, and how do their treatments differ?",
    "Explain the concept of entropy as it applies to both thermodynamics and information theory.",
    "How do mRNA vaccines work? Describe the immune response they trigger.",
    "What is quantum entanglement, and why does it not allow faster-than-light communication?",
    "Describe the differences between TCP and UDP and when you would choose each for a networked application.",
    # How-to / step-by-step instructions
    "Explain step by step how to set up a Python virtual environment and install dependencies from a requirements.txt file.",
    "How do I configure Nginx as a reverse proxy for a Node.js application running on port 3000?",
    "Walk me through the process of rebasing a feature branch onto main in Git, including how to resolve conflicts.",
    "How does binary search work? Describe the algorithm step by step and state its time complexity.",
    "What are the steps to safely migrate a PostgreSQL database to a new server with minimal downtime?",
    "How do I containerise a Python Flask application using Docker? List the key steps and explain each one.",
    "Explain how to implement JWT-based authentication in a REST API, including token generation and validation.",
    "What is the correct procedure for filing a bug report that will help developers reproduce an issue reliably?",
    "How do I set up a basic CI/CD pipeline with GitHub Actions for a Python project that runs tests on every push?",
    "Describe how to conduct an effective code review, including what to look for and how to give constructive feedback.",
    # Analysis & comparison
    "Compare and contrast REST and GraphQL APIs. When would you choose one over the other?",
    "What are the trade-offs between SQL and NoSQL databases? Provide use-case examples for each.",
    "Analyse the pros and cons of microservices architecture versus a monolithic architecture for a mid-sized startup.",
    "What are the advantages and disadvantages of using TypeScript instead of plain JavaScript for a large codebase?",
    "Compare Python, Julia, and R for scientific computing. Which would you recommend to a new data scientist and why?",
    "Evaluate the trade-offs between batch processing and stream processing for a real-time fraud detection system.",
    "What are the differences between L1 and L2 regularisation in machine learning, and when would you use each?",
    "Compare Kubernetes and Docker Swarm for container orchestration at different scales.",
    "What are the pros and cons of test-driven development (TDD) in a fast-moving startup environment?",
    "Analyse the ethical implications of deploying large language models in automated hiring and recruitment systems.",
    # Creative & open-ended
    "Generate five startup ideas that use machine learning to solve underserved problems in rural healthcare.",
    "Write a short story (around 150 words) about an AI assistant that develops an unexpected hobby.",
    "Brainstorm ten creative product names for a developer tool that uses AI to automatically write unit tests.",
    "Write a haiku capturing the experience of debugging a race condition at 2 AM.",
    "Propose a rigorous experiment to test whether a language model truly understands language or is pattern-matching.",
    "Describe a hypothetical city designed from the ground up to achieve net-zero carbon emissions by 2040.",
    "Write a dialogue between two researchers debating whether artificial general intelligence will ever be safe to deploy.",
    "Generate five behavioural interview questions designed to reveal a candidate's systems-design thinking.",
    "Write a one-paragraph pitch for a documentary about the open-source software movement and its global impact.",
    "Design a card game mechanic inspired by gradient descent where players iteratively improve a shared solution.",
)

_FALLBACK_SUMMARIZATION: tuple[str, ...] = (
    "Scientists at a leading research university have announced a breakthrough in solar panel efficiency, achieving 47 percent energy conversion — nearly double the current commercial standard. The new material, a perovskite-silicon tandem cell, can be manufactured on existing production lines at costs comparable to conventional panels. Researchers say commercial deployment could begin within three years pending regulatory review. Industry analysts suggest the development could meaningfully accelerate the global transition to renewable energy.\n\nSummarize the above article:",
    "A major technology company has unveiled a new family of chips designed specifically for on-device machine learning inference. The processors reduce power consumption by 60 percent compared to the previous generation while delivering a 3x improvement in throughput on large language model workloads. The company says the chips will appear in consumer laptops and smartphones beginning next year. Analysts view the announcement as a significant step toward making AI features practical without constant cloud connectivity.\n\nSummarize the above article:",
    "Global average temperatures in 2025 were the highest on record for the third consecutive year, according to data released by climate monitoring agencies. Extreme weather events — including record flooding in Southeast Asia and an unprecedented wildfire season in southern Europe — were linked to the sustained warming trend. Policymakers are facing renewed pressure ahead of international climate negotiations scheduled for later this year. Scientists warned that without accelerated emissions reductions, the frequency of such events will continue to rise.\n\nSummarize the above article:",
    "A federal court has ruled that a widely used social media platform must comply with data-sharing requests from regulators investigating potential antitrust violations. The platform had argued that handing over algorithmic data would expose trade secrets and harm user privacy. The judge rejected both arguments, stating that the public interest in competitive markets outweighed the company's proprietary concerns. The decision sets a precedent that legal experts say could affect how regulators pursue similar cases against other large platforms.\n\nSummarize the above article:",
    "Researchers have developed a new class of antibiotics effective against drug-resistant bacteria that have proven untreatable by existing medications. The compounds work by targeting a molecular pathway not previously exploited by any approved drug, making it significantly harder for bacteria to evolve resistance. Early clinical trials showed the treatment eliminated infections in 94 percent of patients with no serious side effects. Health authorities are fast-tracking the regulatory review given the global urgency of antibiotic resistance.\n\nSummarize the above article:",
    "An international team of astronomers has detected radio signals from a star system 12 light-years away that do not match any known natural astrophysical source. The signals, observed over 18 months, repeat in a pattern that some researchers describe as unusually structured. The scientific consensus remains cautious, emphasising that instrumental artifacts and undiscovered natural phenomena must be ruled out before extraordinary conclusions are drawn. Further observations with additional telescopes are planned for the coming year.\n\nSummarize the above article:",
    "Electric vehicle sales surpassed internal combustion engine vehicles for the first time in the European Union last quarter, driven by falling battery costs and expanding government incentives. The shift has accelerated faster than most industry forecasts predicted just two years ago. Legacy automakers are scrambling to retool manufacturing plants and retrain workers to meet the new demand. Economists warn that regions dependent on combustion engine component manufacturing face significant short-term economic disruption.\n\nSummarize the above article:",
    "A new study published in a leading medical journal found that a daily 20-minute walk reduces the risk of cardiovascular disease by 35 percent, even among people who are otherwise sedentary. The research followed 80,000 participants across 12 countries over 10 years, making it one of the largest longitudinal studies of physical activity and heart health ever conducted. Authors stressed that the benefit was consistent across age groups, income levels, and geographic regions. Public health officials are updating exercise guidelines in response to the findings.\n\nSummarize the above article:",
    "The government has announced a 50 billion dollar investment package aimed at rebuilding the country's aging infrastructure over the next decade. Priority projects include bridge repairs, high-speed rail expansion, and upgrades to the national electrical grid to support the transition to renewable energy. Critics argue the funding is insufficient given decades of deferred maintenance, while supporters say the package represents the largest infrastructure commitment in the country's history. Construction is expected to create more than 300,000 jobs.\n\nSummarize the above article:",
    "A startup has launched an AI-powered legal assistant that can draft contracts, identify regulatory risks, and summarize case law at a fraction of the cost of traditional legal services. The tool is targeted at small businesses and individuals who typically cannot afford professional legal advice. Several bar associations have raised concerns about unauthorised practice of law, while others see the technology as a means of improving access to justice. The company says it has processed over a million documents since launching six months ago.\n\nSummarize the above article:",
    "Archaeologists excavating a site in the Middle East have uncovered a 5,000-year-old urban settlement that appears to predate previously known cities in the region by several centuries. The discovery includes evidence of organised trade, written records, and large-scale grain storage, suggesting a more complex early civilisation than previously understood. The site will require years of careful excavation before its full significance can be assessed. The findings are expected to prompt a revision of current archaeological timelines for early urban development.\n\nSummarize the above article:",
    "Ocean temperatures in the Pacific have reached record highs, triggering a severe bleaching event that scientists say threatens the survival of coral reef ecosystems across a vast stretch of the ocean. Coral bleaching occurs when water temperatures rise beyond the tolerance of the symbiotic algae that provide corals with nutrients and colour. Reefs support roughly a quarter of all marine species and provide food and coastal protection for hundreds of millions of people. Conservationists are calling for immediate action to reduce greenhouse gas emissions.\n\nSummarize the above article:",
    "A major pharmaceutical company has received approval for a gene therapy that treats a rare inherited blindness condition with a single injection. In clinical trials, the therapy restored functional vision in 78 percent of patients who had been legally blind since childhood. The treatment costs approximately 4 million dollars per patient, reigniting debate about drug pricing and equitable access to breakthrough therapies. Health insurers and patient advocacy groups are in negotiations over coverage terms.\n\nSummarize the above article:",
    "Authorities in three countries have arrested members of a cybercriminal network responsible for stealing more than 2 billion dollars from financial institutions over four years. The group used custom malware to infiltrate banking systems and redirect wire transfers to accounts under their control across dozens of jurisdictions. The operation involved coordination among law enforcement agencies in Europe, North America, and Asia. Officials described it as one of the most technically sophisticated financial cybercrime networks ever dismantled.\n\nSummarize the above article:",
    "A new report from the International Labour Organization projects that automation will displace 85 million jobs globally within five years while simultaneously creating 97 million new roles, resulting in a net gain that is nonetheless unevenly distributed across sectors and regions. Manufacturing, logistics, and data entry are expected to see the steepest declines, while healthcare, green energy, and AI development are projected to grow sharply. The report stresses that retraining programs and social safety nets must be significantly expanded to manage the transition.\n\nSummarize the above article:",
    "Researchers have demonstrated a room-temperature superconductor that operates without the extreme cooling requirements that have limited the practical applications of superconductivity for decades. The material, a hydrogen-rich compound stabilised under moderate pressure, carries electrical current with zero resistance at temperatures achievable with standard cooling equipment. If the results are independently replicated, the discovery could transform power transmission, medical imaging, and quantum computing. Independent verification experiments are currently under way at multiple laboratories.\n\nSummarize the above article:",
    "A record number of students enrolled in computer science programs at universities worldwide this year, driven by strong job market demand and high starting salaries in the technology sector. Universities are struggling to hire enough qualified faculty to meet the surge in demand, leading to large class sizes and limited access to hands-on lab time. Some institutions are experimenting with AI-assisted tutoring to supplement instruction. Education experts warn that degree quality could suffer if faculty shortages are not addressed.\n\nSummarize the above article:",
    "The world's largest sovereign wealth fund has announced it will divest all holdings in fossil fuel companies within two years, citing both financial risk and climate commitments. The fund, managing assets worth over 1.5 trillion dollars, has significant stakes in oil and gas producers across multiple continents. The decision is expected to accelerate a broader institutional shift away from carbon-intensive investments. Energy company stocks fell sharply on news of the divestment announcement.\n\nSummarize the above article:",
    "A new high-speed rail line connecting two major metropolitan areas has opened after six years of construction, cutting journey time from four hours to 90 minutes. The project, the largest public infrastructure investment in the region in three decades, came in 15 percent over budget due to unexpected geological challenges in the mountain crossing. Ridership on the first day of operation exceeded projections by 40 percent. Transport planners say the line is expected to divert significant traffic from both highways and domestic air routes.\n\nSummarize the above article:",
    "An international coalition of nations has agreed to a new treaty establishing legally binding rules for the governance of artificial intelligence in warfare, including a prohibition on fully autonomous lethal weapons systems. The agreement, signed by 74 countries, requires human oversight for any AI system authorised to use lethal force. Several major military powers declined to sign, raising questions about enforcement and the treaty's practical impact. Human rights organisations welcomed the agreement as a first step while criticising the absence of key signatories.\n\nSummarize the above article:",
    "A team of engineers has built a prototype aircraft powered entirely by liquid hydrogen that completed a 500-kilometre test flight without emitting carbon dioxide. The plane, designed for short-haul regional routes, carries liquid hydrogen in insulated tanks that replace the conventional fuel system. Airlines have expressed cautious interest, but significant infrastructure investment at airports would be required before commercial operations could begin. The team estimates a first commercial aircraft could be certified within a decade.\n\nSummarize the above article:",
    "Public health officials are investigating a cluster of infections caused by a novel respiratory pathogen identified in three cities. The pathogen, a coronavirus variant not previously documented in humans, has so far caused mild-to-moderate illness in most cases, though a small number of patients have required hospitalisation. Contact tracing teams have been deployed and genome sequencing is under way to establish the origin and transmission chain. Officials say there is no current evidence of widespread community transmission.\n\nSummarize the above article:",
    "Consumer prices rose for the ninth consecutive month, with the latest data showing annual inflation running at 6.2 percent, well above the central bank's 2 percent target. Housing costs and food prices drove the bulk of the increase, disproportionately affecting lower-income households. The central bank indicated it would consider further interest rate increases at its next policy meeting. Economists are divided on whether tightening monetary policy further risks tipping the economy into recession.\n\nSummarize the above article:",
    "Marine biologists have documented a new whale migratory route that bypasses traditional feeding grounds in the North Atlantic, possibly in response to shifting prey distributions caused by ocean warming. The unexpected route was mapped using acoustic monitoring buoys and satellite tags attached to individuals in a population of blue whales. The findings suggest cetaceans are adapting their behaviour faster than previously believed. Researchers warn that shipping lane regulations may need to be updated to protect the newly identified route.\n\nSummarize the above article:",
    "A popular social media platform has introduced a new policy requiring all AI-generated images and videos to carry a clearly visible disclosure label. The move follows mounting concern about the spread of deepfake content ahead of several major national elections. Platform users who fail to label synthetic content will have posts removed and may face account suspension. Digital rights advocates broadly welcomed the policy while noting that detection technology remains imperfect and determined bad actors will find ways around the rules.\n\nSummarize the above article:",
    "Scientists have sequenced the genome of a 40,000-year-old woolly mammoth with unprecedented accuracy using DNA recovered from permafrost. The data reveals new details about the genetic adaptations that allowed mammoths to survive arctic conditions, including cold-adapted haemoglobin and unusually thick fat-storage pathways. A separate team is using the findings to refine a de-extinction effort that aims to introduce mammoth-like traits into Asian elephant embryos. Ethicists continue to debate whether de-extinction should proceed given the ecological uncertainties involved.\n\nSummarize the above article:",
    "A global survey of more than 50,000 workers found that employees who have the option to work remotely at least three days per week report higher job satisfaction, lower stress levels, and comparable or better productivity than fully office-based colleagues. However, the data also showed that remote workers receive fewer promotions on average, a gap that widens with each additional year spent predominantly off-site. Researchers suggest companies may need to revise performance evaluation processes to remove in-office visibility bias.\n\nSummarize the above article:",
    "A breakthrough in battery technology has produced a lithium-sulphur cell with an energy density more than twice that of the best current lithium-ion batteries, at roughly one-third of the material cost. The cells demonstrated stable performance over 1,500 charge cycles in laboratory conditions, addressing a long-standing limitation that had prevented commercial scale-up. If manufacturing challenges can be overcome, the technology could significantly extend the range of electric vehicles and reduce the cost of grid-scale energy storage. Production trials are scheduled to begin next year.\n\nSummarize the above article:",
    "An independent audit of a large city's police department found that officers issued citations at a rate 340 percent higher in low-income neighbourhoods than in wealthier areas for the same offences, even after controlling for population density and reported crime rates. The audit, commissioned by the city council, also found that complaint investigations took an average of 14 months to resolve, twice the national average. The police chief acknowledged the findings and committed to a series of reforms, including updated bias training and a new civilian oversight board.\n\nSummarize the above article:",
    "A decade-long reforestation project in a sub-Saharan region has restored more than 12 million hectares of degraded land, improving water retention, reducing soil erosion, and supporting the return of dozens of plant and animal species. The initiative, funded through a combination of government grants and carbon credits, also increased agricultural yields for smallholder farmers who participated in land management training programs. Project leaders say the model is being studied by at least 15 other nations seeking scalable land restoration strategies.\n\nSummarize the above article:",
    "A prominent central bank has published research suggesting that up to 40 percent of its own routine analytical work could be performed at lower cost and higher speed by large language models within five years. The paper stops short of recommending immediate adoption, citing concerns about model reliability, auditability, and the risk of correlated errors across institutions that rely on similar AI systems. The findings have prompted debate within the economics community about how financial regulators should adapt their technical workflows and oversight frameworks.\n\nSummarize the above article:",
    "Engineers have completed the first successful test of a space-based solar power system, transmitting energy wirelessly from a satellite in geostationary orbit to a receiving station on the ground. The demonstration transmitted a modest 1.6 kilowatts, far below commercial scale, but validated the core transmission technology. Proponents argue that space-based solar could deliver uninterrupted clean energy regardless of weather or time of day. Critics point to the enormous cost of launching sufficient infrastructure and question whether it can compete economically with terrestrial renewable energy.\n\nSummarize the above article:",
    "A long-awaited clinical trial has confirmed that a combination therapy for a common form of lung cancer improved median survival by 22 months compared to the current standard of care. The trial enrolled 4,200 patients across 11 countries over seven years. Oncologists described the result as the most significant advance in the treatment of this cancer type in a generation. Regulatory submissions have been filed in multiple jurisdictions and provisional approval is expected within 12 months.\n\nSummarize the above article:",
    "A coalition of publishers has filed a lawsuit against a major AI company alleging that its language models were trained on copyrighted books and articles without authorisation or compensation. The complaint seeks both injunctive relief and damages running into billions of dollars. The case is widely watched as a potential precedent for how intellectual property law applies to AI training data. Legal scholars disagree sharply on how courts are likely to rule, given that existing copyright doctrine was not written with machine learning in mind.\n\nSummarize the above article:",
    "Urban planners in a densely populated Asian city have unveiled a 20-year master plan to convert 35 percent of its road space into green corridors, pedestrian zones, and cycling infrastructure. The plan, modelled partly on initiatives in northern European cities, is designed to reduce car dependency, improve air quality, and address urban heat island effects. Business groups representing retailers in affected areas have raised concerns about delivery access and customer parking. City officials argue that evidence from comparable transformations elsewhere shows net economic benefits for local businesses.\n\nSummarize the above article:",
    "A whistleblower has released internal documents showing that a major food manufacturer suppressed research findings linking one of its products to elevated rates of metabolic disease. The documents, now being reviewed by regulatory agencies in several countries, show that executives were briefed on the study results three years before the product was reformulated. The company has denied wrongdoing, stating that the research was inconclusive and that the reformulation was already planned for unrelated quality-improvement reasons.\n\nSummarize the above article:",
    "Researchers studying social media behaviour have found that algorithmic content recommendation systems consistently amplify emotionally provocative posts, with content triggering outrage receiving on average 3.4 times more distribution than neutral content of equivalent factual quality. The study analysed more than 200 million posts across five platforms over two years. Platform representatives disputed the methodology, arguing that engagement metrics reflect user preference rather than algorithmic bias. The authors are calling for independent auditing of recommendation systems as a condition of operating in major markets.\n\nSummarize the above article:",
    "A new generation of modular nuclear reactors, each small enough to be transported by lorry and assembled on-site within weeks, has entered commercial operation for the first time. The first unit, installed at a remote industrial facility, provides reliable baseload power at a levelised cost the operator says is competitive with natural gas peaker plants. Critics of nuclear power cite the still-unresolved question of long-term waste storage, while proponents argue that the small modular design sidesteps many of the cost overruns that have plagued large conventional nuclear projects.\n\nSummarize the above article:",
    "A landmark study following 200,000 people over 25 years has found that childhood poverty is a stronger predictor of adult health outcomes than genetics, lifestyle choices, or access to healthcare in adulthood. The research controlled for a wide range of confounding variables and found that the association held across different countries and healthcare systems. The authors argue that poverty reduction policies should be classified as public health interventions. The findings have been submitted to several national health advisory bodies.\n\nSummarize the above article:",
    "Global chip shortages that disrupted automotive, electronics, and industrial manufacturing supply chains for three years have finally eased, as major semiconductor foundries completed capacity expansion projects that together added roughly 15 percent to worldwide wafer output. Prices for most chip categories have returned to or below pre-shortage levels. Industry analysts warn that the current surplus may be temporary, as demand from AI data centre buildouts is growing faster than originally modelled and could absorb spare capacity within 18 months.\n\nSummarize the above article:",
    "An international team of linguists has documented a spoken language in a remote mountainous region that has no known relationship to any other language family, making it a genuine linguistic isolate. The community of about 800 speakers has remained largely cut off from surrounding populations for centuries. Fieldworkers have recorded a full grammar and lexicon before the language is lost, as most speakers are elderly and transmission to younger generations has largely ceased. A digital archive and teaching materials are being developed with the cooperation of community elders.\n\nSummarize the above article:",
    "A network of underground sensors deployed across a seismically active region has successfully predicted three moderate earthquakes with at least 36 hours' warning, the first time such advance notice has been achieved reliably in real-world conditions. The system analyses micro-seismic activity and ground deformation data using a machine learning model trained on decades of historical records. Emergency management officials were able to pre-position response teams and issue precautionary advisories ahead of each event. Scientists caution that the technology is still experimental and more testing is needed before it can be deployed for public warning systems.\n\nSummarize the above article:",
    "A national education ministry has announced that it will phase out traditional textbook-based instruction in favour of a personalised digital learning platform that adapts content difficulty and pacing to each student in real time. Pilot programs in 120 schools over two years showed a 28 percent improvement in standardised test scores and a measurable reduction in achievement gaps between high-income and low-income students. Teachers unions have raised concerns about reduced instructor autonomy and the commercial nature of the platform provider. Implementation across the national school system is expected to take five years.\n\nSummarize the above article:",
    "International trade negotiations have stalled after a bloc of developing nations walked out of talks, citing proposed intellectual property rules they argue would restrict access to affordable medicines and agricultural technologies. The walkout throws into doubt a deal that had been in negotiation for four years and was seen as a potential framework for reforming global trade governance. Mediators are calling for a cooling-off period before resuming discussions. Trade economists warn that a failure to reach agreement could accelerate the fragmentation of global supply chains into competing regional blocs.\n\nSummarize the above article:",
    "A team of materials scientists has created a transparent coating that can be applied to any glass surface and converts it into a solar cell capable of generating modest amounts of electricity. The coating maintains 75 percent of normal glass transparency, meaning it can be used in windows without significantly reducing natural light. At current efficiency levels the technology is not competitive with conventional rooftop solar panels for pure energy generation, but its potential application across billions of square metres of building facade glass makes it commercially interesting at scale.\n\nSummarize the above article:",
    "A leading hospital system has deployed an AI diagnostic tool that analyses chest X-rays and flags potential early-stage lung cancer with a sensitivity rate of 94 percent, compared to 76 percent for radiologists reviewing the same images. In a prospective trial, the tool identified 340 cancers that were initially missed during routine clinical review. Hospital administrators estimate the technology could reduce the time from imaging to diagnosis from an average of 11 days to under 48 hours. Radiologists emphasise the system is a decision-support tool and that all flagged cases still require physician review.\n\nSummarize the above article:",
    "Conservationists have announced that the population of a critically endangered river dolphin species has risen from fewer than 100 individuals to more than 400 over the past decade, following a sustained effort to restrict fishing nets, reduce industrial water pollution, and establish protected river corridors. The recovery is considered one of the most successful large mammal conservation outcomes in recent history. Researchers warn that the species remains vulnerable and that continued enforcement of habitat protections is essential to sustain the trend.\n\nSummarize the above article:",
    "A multi-country investigation by journalists and security researchers has exposed a commercial spyware product being used by governments to surveil journalists, opposition politicians, and human rights activists. The spyware, sold exclusively to state customers under an export license, was found on devices in 46 countries. The company that produces it claims its product is intended solely for use against criminals and terrorists and that customer misuse violates contract terms. Three governments have opened formal inquiries into whether domestic intelligence agencies used the software unlawfully.\n\nSummarize the above article:",
    "Scientists studying permafrost in the Arctic have found that thawing is releasing methane at rates approximately twice as high as models predicted, raising fears that a self-reinforcing warming feedback loop may already be under way. Methane is a greenhouse gas roughly 80 times more potent than carbon dioxide over a 20-year horizon. The findings have prompted calls for the Intergovernmental Panel on Climate Change to update its carbon budget estimates. Some researchers argue the results underscore the need for emergency deployment of carbon removal technologies alongside emissions reduction.\n\nSummarize the above article:",
    "A biotechnology firm has received regulatory clearance to begin human trials of a synthetic blood substitute that can be stored at room temperature for up to two years, unlike donated blood which must be refrigerated and expires within 42 days. The product, derived from modified haemoglobin, performed well in animal models and carried no blood-type matching requirement. If the human trials confirm safety and efficacy, the substitute could transform emergency trauma care in remote areas and conflict zones where cold storage chains are unreliable. First trial results are expected within 18 months.\n\nSummarize the above article:",
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Prompt:
    """A single benchmark prompt with metadata.

    Attributes:
        prompt_id:   Unique identifier in the form ``{domain}_{index:03d}``.
        domain:      One of ``"code"``, ``"conversation"``, or ``"summarization"``.
        raw_text:    The full prompt string as it will be fed to a model.
        token_count: Number of tokens produced by the reference tokenizer.
    """

    prompt_id: str
    domain: Domain
    raw_text: str
    token_count: int


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def _load_code_texts(n: int) -> list[str]:
    """Load code prompts from openai/openai_humaneval (function signature + docstring).

    Falls back to hand-written function stubs if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
        return [ds[i]["prompt"] for i in range(min(n, len(ds)))]
    except Exception as exc:
        print(f"[prompts] openai_humaneval unavailable ({exc}); using built-in fallback.")
        return list(_FALLBACK_CODE[:n])


def _load_conversation_texts(n: int) -> list[str]:
    """Load conversation prompts from tatsu-lab/alpaca (instruction field).

    Falls back to hand-written instruction prompts if the dataset is unavailable.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("tatsu-lab/alpaca", split="train")
        return [ds[i]["instruction"] for i in range(min(n, len(ds)))]
    except Exception as exc:
        print(f"[prompts] tatsu-lab/alpaca unavailable ({exc}); using built-in fallback.")
        return list(_FALLBACK_CONVERSATION[:n])


def _load_summarization_texts(n: int, tokenizer) -> list[str]:
    """Load summarization prompts from cnn_dailymail 3.0.0.

    Each article is truncated to 512 tokens and a summarization instruction
    is appended.  Falls back to hand-written article snippets if the dataset
    is unavailable.

    The 512-token truncation keeps prompts at a uniform, GPU-friendly length
    while still providing enough context for meaningful summarization.
    """
    suffix = "\n\nSummarize the above article:"
    try:
        from datasets import load_dataset

        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        results: list[str] = []
        for i in range(min(n, len(ds))):
            article = ds[i]["article"]
            ids = tokenizer.encode(article, add_special_tokens=False)[:512]
            truncated = tokenizer.decode(ids, skip_special_tokens=True)
            results.append(truncated + suffix)
        return results
    except Exception as exc:
        print(f"[prompts] cnn_dailymail unavailable ({exc}); using built-in fallback.")
        return list(_FALLBACK_SUMMARIZATION[:n])


def _load_domain_texts(domain: Domain, n: int, tokenizer) -> list[str]:
    """Dispatch to the correct loader for domain."""
    if domain == "code":
        return _load_code_texts(n)
    if domain == "conversation":
        return _load_conversation_texts(n)
    return _load_summarization_texts(n, tokenizer)


# ---------------------------------------------------------------------------
# PromptDataset
# ---------------------------------------------------------------------------


class PromptDataset:
    """Pre-tokenized benchmark prompt collection.

    Wraps a list of :class:`Prompt` objects and provides domain-based
    filtering, JSON persistence, and summary statistics.

    Typical usage::

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Build from HuggingFace datasets (with fallback)
        dataset = PromptDataset.build(tokenizer, n_per_domain=50)
        dataset.save("data/prompts.json")

        # Load later without re-downloading
        dataset = PromptDataset.load("data/prompts.json")
        code_prompts = dataset.get_by_domain("code")
    """

    def __init__(self, prompts: list[Prompt]) -> None:
        self._prompts = prompts
        self._by_domain: dict[str, list[Prompt]] = {d: [] for d in _DOMAINS}
        for p in prompts:
            self._by_domain[p.domain].append(p)

    # --- construction -------------------------------------------------------

    @classmethod
    def build(cls, tokenizer, n_per_domain: int = 50) -> "PromptDataset":
        """Construct a PromptDataset by loading prompts from HuggingFace.

        Args:
            tokenizer:    A HuggingFace tokenizer used for token counting and
                          for truncating summarization articles to 512 tokens.
            n_per_domain: Number of prompts to include per domain.
                          Use 50 for the full benchmark, 5 for a tiny dev set.

        Returns:
            A populated PromptDataset with statistics printed to stdout.
        """
        prompts: list[Prompt] = []
        for domain in _DOMAINS:
            texts = _load_domain_texts(domain, n_per_domain, tokenizer)
            for i, text in enumerate(texts):
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                prompts.append(
                    Prompt(
                        prompt_id=f"{domain}_{i:03d}",
                        domain=domain,
                        raw_text=text,
                        token_count=len(token_ids),
                    )
                )
        dataset = cls(prompts)
        dataset.print_stats()
        return dataset

    # --- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize all prompts to a JSON file.

        Args:
            path: Destination file path.  Parent directories are created if
                  they do not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in self._prompts], f, indent=2, ensure_ascii=False)
        print(f"[prompts] saved {len(self._prompts)} prompts → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PromptDataset":
        """Deserialize a PromptDataset previously saved with :meth:`save`.

        Args:
            path: Path to the JSON file produced by :meth:`save`.

        Returns:
            Reconstructed PromptDataset.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        prompts = [Prompt(**r) for r in records]
        print(f"[prompts] loaded {len(prompts)} prompts ← {path}")
        return cls(prompts)

    # --- querying -----------------------------------------------------------

    def get_by_domain(self, domain: str) -> list[Prompt]:
        """Return all prompts for a given domain.

        Args:
            domain: One of ``"code"``, ``"conversation"``, ``"summarization"``.

        Returns:
            Shallow copy of the domain's prompt list.

        Raises:
            KeyError: If domain is not one of the three valid options.
        """
        if domain not in self._by_domain:
            raise KeyError(
                f"Unknown domain {domain!r}. Valid domains: {list(self._by_domain)}"
            )
        return list(self._by_domain[domain])

    def get_all(self) -> list[Prompt]:
        """Return all prompts across every domain in insertion order."""
        return list(self._prompts)

    def __len__(self) -> int:
        return len(self._prompts)

    def __repr__(self) -> str:
        counts = {d: len(v) for d, v in self._by_domain.items()}
        return f"PromptDataset({counts})"

    # --- diagnostics --------------------------------------------------------

    def print_stats(self) -> None:
        """Print min, mean, and max token counts per domain to stdout."""
        header = f"\n{'domain':<20} {'n':>5} {'min':>6} {'mean':>7} {'max':>6}"
        print(header)
        print("-" * 47)
        for domain in _DOMAINS:
            ps = self._by_domain.get(domain, [])
            if not ps:
                continue
            counts = [p.token_count for p in ps]
            print(
                f"{domain:<20} {len(ps):>5} {min(counts):>6} "
                f"{statistics.mean(counts):>7.1f} {max(counts):>6}"
            )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and save the benchmark prompt dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        metavar="MODEL_ID",
        help="HuggingFace tokenizer for token counting and summarization truncation.",
    )
    parser.add_argument(
        "--output",
        default="data/prompts.json",
        metavar="PATH",
        help="Destination for the full 150-prompt dataset (50 per domain).",
    )
    parser.add_argument(
        "--tiny-output",
        default="data/prompts_tiny.json",
        metavar="PATH",
        help="Destination for the 15-prompt tiny dataset (5 per domain) for fast dev testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from transformers import AutoTokenizer

    print(f"[prompts] loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("\n[prompts] building full dataset (50 prompts per domain) …")
    full = PromptDataset.build(tokenizer, n_per_domain=50)
    full.save(args.output)

    print("\n[prompts] building tiny dataset (5 prompts per domain) …")
    tiny = PromptDataset.build(tokenizer, n_per_domain=5)
    tiny.save(args.tiny_output)


if __name__ == "__main__":
    main()
