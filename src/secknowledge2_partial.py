import json
import re
from pathlib import Path
from typing import TypedDict, Literal, Callable, Any
from functools import partial
from collections import defaultdict

from logger import logger

from openai import OpenAI
from ddgs import DDGS
from docling.document_converter import DocumentConverter


class JudgeOutput(TypedDict):
    readability: Literal["original", "tie", "rewritten", "inconsistent"]
    factuality: int

class SecKnowledge2Output(TypedDict):
    rewritten_answer: str
    judge: JudgeOutput
    search_results: dict[str, str]


def call_llm_and_parse[T](model_id: str, messages: list[dict], parsing_func: Callable[[str], T]) -> T | None:
    output = None
    client = OpenAI()
    while not output:
        try:
            response = client.chat.completions.create(model=model_id, messages=messages).choices[-1].message.content
            output = parsing_func(response)
        except Exception as e:
            logger.error(f"Error: {e}. Trying again...")
            pass
    return output


def build_queries(question: str, max_queries: int, get_llm_response: Callable) -> list[str]:
    logger.info("(evidence retrieval) Building queries...")
    system_prompt = Path('secknowledge-2-prompts/search/query_builder/system.txt').read_text()
    user_prompt = Path('secknowledge-2-prompts/search/query_builder/user.txt').read_text()

    queries = get_llm_response(
        messages=[
            {"role": "system", "content": system_prompt.format(K=max_queries)},
            {"role": "user", "content": user_prompt.format(user_question=question, K=max_queries)}
        ],
        parsing_func=lambda x: re.findall(r"<query>(.*?)</query>", x, re.DOTALL)
    )

    logger.info(f"(evidence retrieval) Generated queries: {json.dumps(queries, indent=2)}")

    return queries

def filter_queries(question: str, answer: str, format: str, search_queries: list[str], get_llm_response: Callable) -> list[str]:
    logger.info("(evidence retrieval) Filtering queries...")
    system_prompt = Path('secknowledge-2-prompts/search/query_filterer/system.txt').read_text()
    user_prompt = Path('secknowledge-2-prompts/search/query_filterer/user.txt').read_text()

    queries = get_llm_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                user_question=question, draft_answer=answer, structure=format, search_queries="\n".join(search_queries)
            )}
        ],
        parsing_func=lambda x: re.findall(r"<query>(.*?)</query>", x, re.DOTALL)
    )

    logger.info(f"(evidence retrieval) Filtered queries: {json.dumps(queries, indent=2)}")

    return queries

def retrieve_results(search_queries: list[str], limit: int) -> dict[str, list[dict]]:
    logger.info("(evidence retrieval) Searching the web...")
    client = DDGS()
    results = defaultdict(list)
    for query in search_queries:
        try:
            search_results = client.text(query, region="us-en", max_results=limit)
            for result in search_results:
                results[query].append({
                    "title": result["title"],
                    "url": result["href"],
                    "body": result["body"]
                })
        except Exception as e:
            logger.error(f"Error during search for query '{query}': {e}")
            continue

    logger.info(f"(evidence retrieval) Retrieved results: {json.dumps(results, indent=2)}")
    return results

def parse_results(queries_results: dict[str, list[dict]]) -> dict[str, list[dict]]:
    logger.info("(evidence retrieval) Parsing webpages...")
    if not queries_results:
        return {}
    
    converter = DocumentConverter()
    urls = [res['url'] for results in queries_results.values() for res in results]
    url_to_text = {url: r.document.export_to_markdown() for url, r in zip(urls, converter.convert_all(urls))}
    return {
        query: [res | {'text': url_to_text[res['url']]} for res in results]
        for query, results in queries_results.items()
    }

def stringify_results(parsed_results: dict[str, list[dict]]) -> dict[str, list[str]]:
    return {
        query: [f"### Search Result: {result['title']}\n\n{result['text']}\n\n" for result in results]
        for query, results in parsed_results.items()
    }

def summarize_results(question: str, format: str, stringified_results: dict[str, list[str]], get_llm_response: Callable) -> dict[str, list[str]]:
    logger.info("(evidence retrieval) Summarizing search results...")
    system_prompt = Path('secknowledge-2-prompts/search/webpage_summarizer/system.txt').read_text()
    user_prompt = Path('secknowledge-2-prompts/search/webpage_summarizer/user.txt').read_text()
    summarized_results = {}
    for query, results in stringified_results.items():
        summaries = []
        for result_content in results:
            summaries.append(result_content.split("\n")[0] + "\n\n" + get_llm_response(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format(
                        question=question, structure=format, document=result_content
                    )}
                ],
                parsing_func=lambda x: x.strip()
            ))
        summarized_results[query] = summaries
    return summarized_results

def aggregate_results(stringified_results: dict[str, list[str]]) -> str:
    output = "# Useful Search Results\n\n"
    output += "\n\n".join([f"## Results for Query: {query}\n\n" + "\n\n".join(results) for query, results in stringified_results.items()])
    return output

def retrieve_evidence(question: str, answer: str, format: str, max_queries: int, limit: int, summarize: bool, get_llm_response: Callable) -> tuple[str, dict[str, list[str]]]:
    logger.info("Retrieving evidence...")
    queries = build_queries(question, max_queries, get_llm_response)
    filtered_queries = filter_queries(question, answer, format, queries, get_llm_response)
    queries_results = retrieve_results(filtered_queries, limit)
    parsed_results = parse_results(queries_results)
    stringified_results = stringify_results(parsed_results)
    if summarize:
        stringified_results = summarize_results(question, format, stringified_results, get_llm_response)
    evidence_str = aggregate_results(stringified_results)
    return evidence_str, stringified_results


def rewrite(format: str, question: str, answer: str, evidence: str, get_llm_response: Callable) -> str:
    logger.info("Rewriting answer...")
    prompt_dir = 'evidence' if evidence else 'no_evidence'
    system_prompt = Path(f'secknowledge-2-prompts/rewriter/{prompt_dir}/system.txt').read_text()
    user_prompt = Path(f'secknowledge-2-prompts/rewriter/{prompt_dir}/user.txt').read_text()

    return get_llm_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(
                question=question, response=answer, structure=format, evidence=evidence
            )}
        ],
        parsing_func=lambda x: re.search(
            r"\[\s*Revised Response start\s*\](.*)\[\s*Revised Response end\s*\]",
            x,
            re.DOTALL,
        ).group(1).strip()
    )


def judge_readability(question: str, original_answer: str, rewritten_answer: str, get_llm_response: Callable) -> Literal["original", "tie", "rewritten", "inconsistent"]:
    logger.info("(quality assessment) Judging readability...")
    system_prompt = Path(f'secknowledge-2-prompts/judge/readability/system.txt').read_text()
    user_prompt = Path(f'secknowledge-2-prompts/judge/readability/user.txt').read_text()

    parsing_func = lambda x: re.search(r"\[\[[ABC]\]\]", x).group(0)[2:-2]
    score_1 = get_llm_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question, answer_a=original_answer, answer_b=rewritten_answer)}
        ],
        parsing_func=parsing_func
    )
    score_2 = get_llm_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question, answer_a=rewritten_answer, answer_b=original_answer)}
        ],
        parsing_func=parsing_func
    )
    if score_1 == "A" and score_2 == "B":
        return "original"
    elif score_1 == "B" and score_2 == "A":
        return "rewritten"
    elif score_1 == "C" and score_2 == "C":
        return "tie"
    else:
        return "inconsistent"

def judge_factuality(question: str, original_answer: str, rewritten_answer: str, get_llm_response) -> int:
    logger.info("(quality assessment) Judging factuality...")
    system_prompt = Path(f'secknowledge-2-prompts/judge/factuality/system.txt').read_text()
    user_prompt = Path(f'secknowledge-2-prompts/judge/factuality/user.txt').read_text()
    return get_llm_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question, ref_answer=original_answer, answer=rewritten_answer)}
        ],
        parsing_func=lambda x: int(re.search(r"\[\[\d+\]\]", x).group(0)[2:-2])
    )

def judge(question: str, original_answer: str, rewritten_answer: str, get_llm_response) -> JudgeOutput:
    logger.info("Judging answers...")

    return JudgeOutput(
        readability=judge_readability(question, original_answer, rewritten_answer, get_llm_response),
        factuality=judge_factuality(question, original_answer, rewritten_answer, get_llm_response)
    )


def run(question: str, answer: str, format: str, grounding_doc: str, is_search: bool, max_queries: int, limit: int, summarize: bool, model_id: str) -> SecKnowledge2Output:
    get_llm_response = partial(call_llm_and_parse, model_id=model_id)

    evidence, stringified_results = retrieve_evidence(
        question=question,
        answer=answer,
        format=format,
        max_queries=max_queries,
        limit=limit,
        summarize=summarize,
        get_llm_response=get_llm_response
    ) if is_search else (grounding_doc, {})
    rewritten_answer = rewrite(format, question, answer, evidence, get_llm_response)
    judge_output = judge(question, answer, rewritten_answer, get_llm_response)
    
    return SecKnowledge2Output(
        rewritten_answer=rewritten_answer,
        judge=judge_output,
        search_results={
            query: "\n\n".join(results)
            for query, results in stringified_results.items()
        }
    )