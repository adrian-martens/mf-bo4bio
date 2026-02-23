import mfbo4bio.mfbo_GIBBON as gibbon
import mfbo4bio.mfbo_qLogEI as qlogei
import mfbo4bio.mfbo_qUCB as qucb


def test_entrypoints_expose_run():
    assert callable(qucb.run)
    assert callable(qlogei.run)
    assert callable(gibbon.run)
