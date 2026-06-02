import csv
from pathlib import Path


def make_text(row: dict[str, str]) -> str:
    """1件のエンティティ情報を埋め込み用の1本の文字列にまとめる。"""
    parts = [str(row["name"])]

    description = row.get("description", "")
    if description.strip():
        parts.append(str(description))

    manufacturer = row.get("manufacturer", "")
    if manufacturer.strip():
        parts.append(f"manufacturer: {manufacturer}")

    price = row.get("price", "")
    if price.strip():
        parts.append(f"price: {price}")

    return " | ".join(parts)


def load_data(file_path: str | Path) -> list[dict[str, str]]:
    """指定した CSV ファイルを UTF-8 で読み込み、行の一覧を返す。"""
    path = Path(file_path)

    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def build_entity_lookup(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """ID から1つの表の行データを引ける辞書を作る。"""
    return {int(row["id"]): row for row in rows}
