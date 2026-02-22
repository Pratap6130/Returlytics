import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


DB_PATH = Path(__file__).resolve().parent / "returns.db"


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str):
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "userid TEXT,"
        "product_category TEXT,"
        "product_price REAL,"
        "order_quantity INTEGER,"
        "user_age INTEGER,"
        "user_gender TEXT,"
        "payment_method TEXT,"
        "shipping_method TEXT,"
        "discount_applied REAL,"
        "prediction INTEGER,"
        "probability REAL,"
        "created_at TEXT)"
    )
    _ensure_column(conn, "predictions", "userid", "TEXT")
    _ensure_column(conn, "predictions", "created_at", "TEXT")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "name TEXT NOT NULL,"
        "userid TEXT NOT NULL UNIQUE,"
        "password_hash TEXT NOT NULL)"
    )
    return conn


def save_prediction(features: Dict[str, Any], prediction: int, probability: float, userid: str | None = None):
    conn = get_connection()
    values = (
        userid,
        features["product_category"],
        float(features["product_price"]),
        int(features["order_quantity"]),
        int(features["user_age"]),
        features["user_gender"],
        features["payment_method"],
        features["shipping_method"],
        float(features["discount_applied"]),
        int(prediction),
        float(probability),
    )
    conn.execute(
        "INSERT INTO predictions (userid, product_category, product_price, "
        "order_quantity, user_age, user_gender, payment_method, "
        "shipping_method, discount_applied, prediction, probability, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        values,
    )
    conn.commit()
    conn.close()


def get_predictions_by_userid(userid: str, limit: int = 100) -> list[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.execute(
        "SELECT id, userid, product_category, product_price, order_quantity, "
        "user_age, user_gender, payment_method, shipping_method, discount_applied, "
        "prediction, probability, created_at "
        "FROM predictions WHERE userid = ? "
        "ORDER BY id DESC LIMIT ?",
        (userid, int(limit)),
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def create_user(name: str, userid: str, password_hash: str) -> bool:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (name, userid, password_hash) VALUES (?, ?, ?)",
            (name, userid, password_hash),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user_by_userid(userid: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.execute(
        "SELECT id, name, userid, password_hash FROM users WHERE userid = ?",
        (userid,),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "id": row[0],
        "name": row[1],
        "userid": row[2],
        "password_hash": row[3],
    }
