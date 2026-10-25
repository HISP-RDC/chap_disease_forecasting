"""ASGI entry point generated from the CHAPKit functional-service contract."""

from chap_disease_forecasting.chapkit_adapter import app as app

if __name__ == "__main__":
    from chapkit.api import run_app

    run_app("main:app", reload=False, port=9090)
