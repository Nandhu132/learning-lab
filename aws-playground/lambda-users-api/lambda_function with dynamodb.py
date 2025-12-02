import json
import boto3

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("users")


def build_response(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body)
    }


# Auto-increment ID function
def get_next_id():
    resp = table.update_item(
        Key={"id": "counter"},
        UpdateExpression="SET #v = #v + :inc",
        ExpressionAttributeNames={"#v": "value"},
        ExpressionAttributeValues={":inc": 1},
        ReturnValues="UPDATED_NEW"
    )
    return str(resp["Attributes"]["value"])


def lambda_handler(event, context):
    http_method = event.get("httpMethod", "")
    path = event.get("path", "")
    path_params = event.get("pathParameters") or {}

    # GET /users
    if path.endswith("/users") and http_method == "GET":
        resp = table.scan()
        items = [
            {
                "id": i["id"],
                "name": i.get("name", ""),
                "email": i.get("email", "")
            }
            for i in resp.get("Items", [])
            if i["id"] != "counter"
        ]

        return build_response(200, {"users": items})

    # POST /users
    if path.endswith("/users") and http_method == "POST":
        body = json.loads(event.get("body", "{}"))
        name = body.get("name")
        email = body.get("email")

        if not name or not email:
            return build_response(400, {"error": "name and email required"})

        new_id = get_next_id()

        item = {"id": new_id, "name": name, "email": email}
        table.put_item(Item=item)

        ordered_item = {
            "id": item["id"],
            "name": item["name"],
            "email": item["email"]
        }

        return build_response(201, ordered_item)

    # PUT /users/{id}
    if "/users/" in path and http_method == "PUT":
        user_id = path_params.get("id")
        if not user_id:
            return build_response(400, {"error": "User ID required"})

        body = json.loads(event.get("body", "{}"))

        update_expr = []
        expr_values = {}
        expr_names = {}  

        if "name" in body:
            update_expr.append("#n = :n")
            expr_values[":n"] = body["name"]
            expr_names["#n"] = "name"      

        if "email" in body:
            update_expr.append("#e = :e")
            expr_values[":e"] = body["email"]
            expr_names["#e"] = "email"

        if not update_expr:
            return build_response(400, {"error": "Nothing to update"})

        table.update_item(
            Key={"id": user_id},
            UpdateExpression="SET " + ", ".join(update_expr),
            ExpressionAttributeValues=expr_values,
            ExpressionAttributeNames=expr_names
        )

        return build_response(200, {"message": "User updated"})

    # DELETE /users/{id}
    if "/users/" in path and http_method == "DELETE":
        user_id = path_params.get("id")
        if not user_id:
            return build_response(400, {"error": "User ID required"})

        table.delete_item(Key={"id": user_id})
        return build_response(200, {"message": "User deleted"})

    return build_response(404, {"error": "Route not found"})
