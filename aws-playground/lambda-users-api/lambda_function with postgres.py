import json
import psycopg2
import os

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432))
}

def connect_db():
    return psycopg2.connect(**DB_CONFIG)


def build_response(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    http_method = event.get("httpMethod") or ""
    path = event.get("path") or ""
    path_params = event.get("pathParameters") or {}


    #  GET /users
    if path == "/users" and http_method == "GET":
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT id, name, email FROM users;")
        users = [{"id": r[0], "name": r[1], "email": r[2]} for r in cur.fetchall()]
        cur.close()
        conn.close()
        return build_response(200, {"users": users})

    #  POST /users
    if path == "/users" and http_method == "POST":
        body = json.loads(event.get("body", "{}"))
        name = body.get("name")
        email = body.get("email")

        if not name or not email:
            return build_response(400, {"error": "name and email required"})

        conn = connect_db()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id;",
                (name, email),
            )
            user_id = cur.fetchone()[0]
            conn.commit()
        except psycopg2.IntegrityError:
            conn.rollback()
            cur.close()
            conn.close()
            return build_response(400, {"error": "email already exists"})
        cur.close()
        conn.close()

        return build_response(201, {"id": user_id, "name": name, "email": email})

    #  PUT /users/{id}
    if path.startswith("/users/") and http_method == "PUT":
        user_id = path_params.get("id")
        if not user_id:
            return build_response(400, {"error": "User ID required"})

        body = json.loads(event.get("body", "{}"))
        name = body.get("name")
        email = body.get("email")

        if not name and not email:
            return build_response(400, {"error": "Nothing to update"})

        conn = connect_db()
        cur = conn.cursor()

        cur.execute("SELECT id FROM users WHERE id = %s;", (user_id,))
        if not cur.fetchone():
            cur.close()
            conn.close()
            return build_response(404, {"error": "User not found"})

        fields = []
        values = []
        if name:
            fields.append("name = %s")
            values.append(name)
        if email:
            fields.append("email = %s")
            values.append(email)

        values.append(user_id)
        sql = f"UPDATE users SET {', '.join(fields)} WHERE id = %s;"

        try:
            cur.execute(sql, values)
            conn.commit()
        except psycopg2.IntegrityError:
            conn.rollback()
            cur.close()
            conn.close()
            return build_response(400, {"error": "email already exists"})

        cur.close()
        conn.close()
        return build_response(200, {"message": "User updated"})

    #  DELETE /users/{id}
    if path.startswith("/users/") and http_method == "DELETE":
        user_id = path_params.get("id")
        if not user_id:
            return build_response(400, {"error": "User ID required"})

        conn = connect_db()
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id = %s;", (user_id,))
        deleted = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()

        if deleted == 0:
            return build_response(404, {"error": "User not found"})

        return build_response(200, {"message": "User deleted"})

    return build_response(404, {"error": "Route not found"})
