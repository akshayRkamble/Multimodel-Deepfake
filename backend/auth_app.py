import sqlite3
import os

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

BASE_DIR = os.path.dirname(__file__)
DATABASE = os.environ.get("AUTH_DATABASE", os.path.join(BASE_DIR, "auth_users.db"))
MAIN_PROJECT_URL = os.environ.get("MAIN_PROJECT_URL", "http://127.0.0.1:5173/index.html")


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    # Some mounted Windows drives fail SQLite's default rollback journal.
    conn.execute("PRAGMA journal_mode=OFF")
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")


def current_user():
    user_id = session.get("user_id")
    if user_id is None:
        return None

    with get_db() as conn:
        return conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


@app.route("/")
def index():
    return render_template("index.html", user=current_user())


@app.route("/main")
def main_project():
    if not current_user():
        flash("Please log in first.", "error")
        return redirect(url_for("login"))

    return redirect(MAIN_PROJECT_URL)


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user():
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if len(username) < 3:
            flash("Username must be at least 3 characters.", "error")
            return render_template("register.html", username=username)

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html", username=username)

        hashed_pw = generate_password_hash(password)

        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    (username, hashed_pw),
                )
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username is already taken.", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(MAIN_PROJECT_URL)

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        with get_db() as conn:
            user = conn.execute(
                "SELECT id, username, password FROM users WHERE username = ?",
                (username,),
            ).fetchone()

            if user and check_password_hash(user["password"], password):
                session.clear()
                session["user_id"] = user["id"]
                flash("Logged in successfully.", "success")
                return redirect(MAIN_PROJECT_URL)

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index"))


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
