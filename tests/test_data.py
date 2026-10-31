from house_price_lab.data import make_dataset


def test_reproducible():
    assert make_dataset(50).equals(make_dataset(50))
