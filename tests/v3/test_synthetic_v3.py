import numpy as np, pytest
from pacbayes_tsk.data.synthetic_v3 import generate
PROCESSES=['ar2','setar','narma10','mackey_glass','garch','structural_break']
@pytest.mark.parametrize('name',PROCESSES)
def test_reproducible(name):
 a=generate(name,length=100,burn_in=50,seed=123); b=generate(name,length=100,burn_in=50,seed=123)
 assert np.array_equal(a.values,b.values); assert np.array_equal(a.regimes,b.regimes); assert np.isfinite(a.values).all()
def test_break_at_sixty_percent():
 s=generate('structural_break',length=100,burn_in=50,seed=1)
 assert np.flatnonzero(s.regimes==1)[0]==60
def test_garch_df5():
 s=generate('garch',length=20,burn_in=10,seed=1); assert s.metadata['degrees_of_freedom']==5.0
