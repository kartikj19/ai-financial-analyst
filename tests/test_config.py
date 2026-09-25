from src.config import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_TOP_K

def test_rag_config_is_sane():
    assert CHUNK_SIZE > CHUNK_OVERLAP > 0
    assert DEFAULT_TOP_K >= 2
