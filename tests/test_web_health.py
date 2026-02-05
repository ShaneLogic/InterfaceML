from interfaceml.web.app import create_app


def test_health_endpoint():
    app = create_app()
    with app.test_client() as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert data.get("status") == "ok"
