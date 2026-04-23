def test_all():
    from src.data import test as test_data
    from src.models import test as test_models
    test_data()
    test_models()
    print("🚀 All system tests passed!")
