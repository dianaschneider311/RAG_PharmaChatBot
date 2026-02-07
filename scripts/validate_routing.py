import os
import sys
from datetime import datetime

sys.path.insert(0, os.getcwd())

from src.retrieval.rag import _route_query_rule, route_query


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    # Labeled query set: (query, expected_route)
    labeled = [
        ("For REP John, list HCPs with neurology specialty", "pharma"),
        ("What are recent findings about ocrelizumab in multiple sclerosis?", "web"),
        ("Latest FDA safety communication on OCREVUS", "web"),
        ("Which HCPs should I visit for Ocrevus outreach?", "pharma"),
        ("Ocrevus prescribing information liver injury", "web"),
    ]

    lines = []
    lines.append("Routing validation")
    lines.append("===================")

    rule_correct = 0
    llm_correct = 0

    for q, expected in labeled:
        rule = _route_query_rule(q)
        llm = route_query(q)
        rule_ok = (rule == expected)
        llm_ok = (llm == expected)
        rule_correct += int(rule_ok)
        llm_correct += int(llm_ok)
        lines.append(f"Query: {q}")
        lines.append(f"Expected: {expected}")
        lines.append(f"Rule: {rule} ({'OK' if rule_ok else 'NO'})")
        lines.append(f"LLM: {llm} ({'OK' if llm_ok else 'NO'})")
        lines.append("")

    lines.append(f"Rule accuracy: {rule_correct}/{len(labeled)}")
    lines.append(f"LLM accuracy: {llm_correct}/{len(labeled)}")

    report_path = f"logs/routing_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
