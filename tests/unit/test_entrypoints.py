"""The `frag` CLI's arguments and terminal input, eval flags and Chainlit patches."""

import string

import pytest
from chainlit import secret
from chainlit.data.storage_clients.s3 import S3StorageClient

from frag import cli
from frag.ui import chainlit_patches
from tests.fakes import FIXTURE_IMAGE


def test_cli_parses_the_eval_flags():
    args = cli._parser().parse_args(["eval", "--skip-run", "--limit", "2"])  # noqa: SLF001
    assert (args.command, args.skip_run, args.skip_judge, args.limit) == (
        "eval",
        True,
        False,
        2,
    )


@pytest.mark.parametrize("argv", [[], ["serve"], ["eval", "--limit", "x"]])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        cli._parser().parse_args(argv)  # noqa: SLF001


async def test_chat_input_skips_image_paths_that_do_not_exist(monkeypatch, capsys):
    answers = iter(["shoes for this", "/no/such/file.jpg", str(FIXTURE_IMAGE), ""])
    monkeypatch.setattr(cli, "_ask", lambda _prompt: next(answers))
    assert await cli._read_message() == ("shoes for this", [str(FIXTURE_IMAGE)])  # noqa: SLF001
    assert "No such file: /no/such/file.jpg" in capsys.readouterr().out


async def test_chat_input_quits(monkeypatch):
    monkeypatch.setattr(cli, "_ask", lambda _prompt: "QUIT")
    assert await cli._read_message() is None  # noqa: SLF001


def test_cli_passes_the_eval_options_through():
    args = cli._parser().parse_args(  # noqa: SLF001
        ["eval", "--sleep", "1.5", "--run-dir", "data/evaluation_runs/x"]
    )
    assert (args.sleep, args.run_dir) == (1.5, "data/evaluation_runs/x")


async def test_chainlit_patches_are_idempotent_and_close_does_not_await(monkeypatch):
    monkeypatch.setattr(chainlit_patches, "_applied", False)
    monkeypatch.setattr(secret, "chars", secret.chars)
    monkeypatch.setattr(S3StorageClient, "close", S3StorageClient.close)
    chainlit_patches.apply()
    chainlit_patches.apply()
    assert set(secret.chars) <= set(string.ascii_letters + string.digits + "-_")

    class Boto3Client:
        closed = False

        def close(self):  # synchronous, like boto3's
            self.closed = True

    storage = object.__new__(S3StorageClient)
    storage.client = Boto3Client()
    await storage.close()
    assert storage.client.closed
