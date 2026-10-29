"""The viewer RPC client keeps an owner refusal's structured verdict (code + bounded data)."""
import asyncio
import json

import pytest

from hermes_cli.gateway_client import GatewayClient, GatewayRPCError


class _Socket:
    """One scripted owner: answers each request with the frame ``reply(request)`` builds."""

    def __init__(self, reply):
        self.reply = reply
        self.inbox = asyncio.Queue()

    async def send(self, raw):
        frame = self.reply(json.loads(raw))
        if frame is not None:
            await self.inbox.put(json.dumps(frame))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


@pytest.mark.asyncio
async def test_rpc_error_keeps_numeric_code_and_bounded_data():
    errors = {
        'session.mutate': {'code': 4001, 'message': 'invalid_params', 'data': {
            'reason': 'invalid_params', 'fields': ['payload.title'],
            'trace': {'nested': 'remote diagnostics'}, 'blob': 'x' * 5000}},
        'nope': {'code': -32601, 'message': 'unknown method: nope', 'data': {'reason': 'unknown_method'}},
    }
    socket = _Socket(lambda req: {'jsonrpc': '2.0', 'id': req['id'], 'error': errors[req['method']]})
    async with GatewayClient(socket) as client:
        with pytest.raises(GatewayRPCError) as refused:
            await client.rpc('session.mutate', session_id='s')
        assert str(refused.value) == 'invalid_params' and refused.value.code == 4001
        # Scalar facts survive; nested objects and oversized strings never reach the caller.
        assert refused.value.data == {'reason': 'invalid_params', 'fields': ['payload.title']}
        with pytest.raises(GatewayRPCError) as unknown:
            await client.rpc('nope')
        # Free-text messages stay bounded: the owner's reason code, not the raw sentence.
        assert str(unknown.value) == 'unknown_method' and unknown.value.code == -32601


@pytest.mark.asyncio
async def test_compress_waits_past_the_default_rpc_budget(monkeypatch):
    """A canonical ``/compress`` answers only after its summary + commit (minutes on a long
    session). The viewer must not report failure at the 30 s default while the owner, which
    shields the mutation, still commits it. Time is scaled 1000x: 30 s -> 30 ms."""
    from hermes_cli.gateway_mutations import PreparedMutations
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(asyncio, 'wait_for', lambda fut, timeout: real_wait_for(fut, timeout / 1000))
    sent = []

    def reply(req):
        sent.append(req)
        if req['method'] == 'session.resume':
            return {'id': req['id'], 'result': {'revision': 1, 'execution_generation': 1}}

        async def commit():  # a 120 s summary (scaled) then the receipt
            await asyncio.sleep(0.12)  # health: allow HX013 -- the reply must land after the scaled 30 ms default budget expired
            await owner.inbox.put(json.dumps({'id': req['id'], 'result': {'status': 'applied'}}))
        asyncio.ensure_future(commit())

    owner = _Socket(reply)
    async with GatewayClient(owner) as client:
        result = await PreparedMutations().apply(client, 's', 'compress', {})
    assert result == {'status': 'applied'}
    # The budget is a client-side knob, never a wire parameter the closed contract would refuse.
    assert all('_timeout' not in req['params'] for req in sent)


@pytest.mark.asyncio
async def test_an_answer_cut_off_by_a_hang_up_is_answer_lost():
    """The socket drops after the send: the request may have reached the owner, so a caller that
    retries must reuse its identity (``AnswerLost``), unlike a request that never left."""
    from hermes_cli.gateway_client import AnswerLost

    class _HangsUp(_Socket):
        async def __anext__(self):
            await self.inbox.get()  # the request was sent; the socket then closes without answering
            raise StopAsyncIteration

    async with GatewayClient(_HangsUp(lambda req: {'sent': True})) as client:
        with pytest.raises(AnswerLost, match='turn outcome is unknown'):
            await client.rpc('session.create', request_id='r')
