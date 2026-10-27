import chess
import json
import pytest

torch=pytest.importorskip('torch')
from chesslab.encoding import encode_board
from chesslab.material import material_cp
from chesslab.model import TinyEvaluator, load_evaluator, NeuralCP
from chesslab.data import generate_split
from chesslab.train import train, evaluate


def test_normalization_preserves_mover_state_and_board():
    board=chess.Board()
    for move in ['e2e4','a7a6','e4e5','d7d5','e5d6']:
        board.push_uci(move)
    original=board.fen()
    assert board.turn==chess.BLACK
    assert encode_board(board,normalized=True)==encode_board(board.mirror(),normalized=True)
    assert board.fen()==original
    ep=chess.Board('rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3')
    assert encode_board(ep,normalized=True)==encode_board(ep.mirror(),normalized=True)


@pytest.mark.parametrize('normalized',[False,True])
@pytest.mark.parametrize('fen',[
    chess.STARTING_FEN,
    'rnbqkbnr/ppp1pppp/8/3p4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2',
    '4k3/8/8/8/8/8/2NQ4/4K3 b - - 0 1',
])
def test_residual_starts_at_material(normalized,fen):
    board=chess.Board(fen)
    model=TinyEvaluator(width=4,normalized=normalized,residual=True)
    x=torch.tensor([encode_board(board,normalized=normalized)])
    assert float(model(x).item())==pytest.approx(material_cp(board)/100,abs=1e-5)


@pytest.mark.parametrize('normalized,residual,loss',[(True,False,'mse'),(False,True,'mse'),(True,True,'huber')])
def test_configured_checkpoint_roundtrip(tmp_path,normalized,residual,loss):
    t,v,c=(tmp_path/name for name in ['train.jsonl','val.jsonl','net.pt'])
    generate_split(t,v,8,4,31)
    result=train(t,v,c,epochs=1,width=4,normalized=normalized,residual=residual,loss=loss,include_initial=True)
    assert json.loads(json.dumps(result))["loss"] == loss
    model=load_evaluator(str(c))
    assert model.normalized==normalized and model.residual==residual
    assert evaluate(c,v)['neural']==result['selected_validation']
    assert isinstance(NeuralCP(str(c))(chess.Board()),int)
    payload=torch.load(c,weights_only=True)
    payload['encoding_version']='wrong-version'
    torch.save(payload,tmp_path/'broken.pt')
    with pytest.raises(ValueError,match='Incompatible'):
        load_evaluator(str(tmp_path/'broken.pt'))
