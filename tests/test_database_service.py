import os
from unittest.mock import MagicMock, patch
import pytest

from src.utils.database_service import DatabaseService


def test_init_defaults(monkeypatch):
    monkeypatch.delenv("LEMURS_POSTGRES_HOST", raising=False)
    monkeypatch.delenv("LEMURS_POSTGRES_PORT", raising=False)
    monkeypatch.delenv("LEMURS_POSTGRES_DB", raising=False)
    monkeypatch.delenv("LEMURS_POSTGRES_USER", raising=False)
    monkeypatch.delenv("LEMURS_POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("LEMURS_SSH_HOST", raising=False)
    monkeypatch.delenv("LEMURS_SSH_KEY", raising=False)
    monkeypatch.delenv("LEMURS_SSH_KEY_PATH", raising=False)

    service = DatabaseService()
    assert service.host == "localhost"
    assert service.port == 5432
    assert service.ssh_host is None
    assert service.connection is None
    assert service.tunnel is None


def test_init_with_ssh_params():
    service = DatabaseService(
        host="db.internal",
        port=5432,
        ssh_host="bastion.example.com",
        ssh_port=2222,
        ssh_user="sshuser",
        ssh_pkey="~/.ssh/id_rsa",
    )
    assert service.ssh_host == "bastion.example.com"
    assert service.ssh_port == 2222
    assert service.ssh_user == "sshuser"
    assert service.ssh_pkey == "~/.ssh/id_rsa"


def test_init_with_ssh_env_vars(monkeypatch):
    monkeypatch.setenv("LEMURS_SSH_HOST", "ssh.cluster.edu")
    monkeypatch.setenv("LEMURS_SSH_PORT", "22")
    monkeypatch.setenv("LEMURS_SSH_USER", "clusteruser")
    monkeypatch.setenv("LEMURS_SSH_KEY_PATH", "/home/user/.ssh/id_ed25519")

    service = DatabaseService()
    assert service.ssh_host == "ssh.cluster.edu"
    assert service.ssh_user == "clusteruser"
    assert service.ssh_pkey == "/home/user/.ssh/id_ed25519"


def test_connect_fails_without_ssh_host():
    service = DatabaseService(ssh_host=None)
    # Ensure no environment variable overrides
    service.ssh_host = None
    success = service.connect()
    assert success is False
    assert service.connection is None
    assert service.tunnel is None


@patch("src.utils.database_service.SSHTunnelForwarder")
@patch("src.utils.database_service.psycopg2.connect")
def test_ssh_tunnel_connect(mock_connect, mock_tunnel_cls):
    mock_tunnel = MagicMock()
    mock_tunnel.local_bind_port = 54321
    mock_tunnel_cls.return_value = mock_tunnel

    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    service = DatabaseService(
        host="localhost",
        port=5432,
        dbname="your_database",
        user="your_username",
        password="your_password",
        ssh_host="bastion.server.edu",
        ssh_user="testuser",
        ssh_pkey="~/.ssh/test_key"
    )

    success = service.connect()

    assert success is True
    assert service.tunnel == mock_tunnel
    mock_tunnel.start.assert_called_once()
    mock_connect.assert_called_once_with(
        host="127.0.0.1",
        port=54321,
        dbname="your_database",
        user="your_username",
        password="your_password"
    )


@patch("src.utils.database_service.SSHTunnelForwarder")
@patch("src.utils.database_service.psycopg2.connect")
def test_disconnect_cleans_up_connection_and_tunnel(mock_connect, mock_tunnel_cls):
    mock_tunnel = MagicMock()
    mock_tunnel.local_bind_port = 54321
    mock_tunnel_cls.return_value = mock_tunnel

    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    service = DatabaseService(
        ssh_host="bastion.server.edu",
    )
    service.connect()
    assert service.connection is not None
    assert service.tunnel is not None

    service.disconnect()
    mock_conn.close.assert_called_once()
    mock_tunnel.stop.assert_called_once()
    assert service.connection is None
    assert service.tunnel is None


@patch("src.utils.database_service.SSHTunnelForwarder")
@patch("src.utils.database_service.psycopg2.connect")
def test_context_manager(mock_connect, mock_tunnel_cls):
    mock_tunnel = MagicMock()
    mock_tunnel.local_bind_port = 54321
    mock_tunnel_cls.return_value = mock_tunnel

    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    with DatabaseService(ssh_host="bastion.server.edu") as db:
        assert db.connection is not None
        assert db.tunnel is not None

    mock_conn.close.assert_called_once()
    mock_tunnel.stop.assert_called_once()
    assert db.connection is None
    assert db.tunnel is None


def test_init_with_raw_ssh_key_env_var(monkeypatch):
    test_key = "-----BEGIN OPENSSH PRIVATE KEY-----\ntest\n-----END OPENSSH PRIVATE KEY-----"
    monkeypatch.setenv("LEMURS_SSH_KEY", test_key)
    monkeypatch.setenv("LEMURS_SSH_HOST", "ssh.host.edu")

    service = DatabaseService()
    assert service.ssh_host == "ssh.host.edu"
    assert service.ssh_key == test_key


def test_resolve_ssh_pkey_from_raw_string():
    dummy_ed25519 = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n"
        "QyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWwAAAKiL2tqii9ra\n"
        "ogAAAAtzc2gtZWQyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWw\n"
        "AAAEB49w67OFQKpqCG3y3uXEepz1YBUxWoZM0iYebm4Me8bbYImqUhbJ/PqdJ2sGhEdicq\n"
        "eC1V2n25ghnrRF8h2chbAAAAImtoaWNrZXlAS2V2aW5zLU1hY0Jvb2stUHJvLTQubG9jYW\n"
        "wBAgM=\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    service = DatabaseService(ssh_key=dummy_ed25519)
    pkey = service._resolve_ssh_pkey()
    assert pkey is not None
    assert pkey.get_name() == "ssh-ed25519"


def test_resolve_ssh_pkey_from_bare_base64_string():
    # Only the key letters, without -----BEGIN / -----END
    bare_letters = (
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW\n"
        "QyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWwAAAKiL2tqii9ra\n"
        "ogAAAAtzc2gtZWQyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWw\n"
        "AAAEB49w67OFQKpqCG3y3uXEepz1YBUxWoZM0iYebm4Me8bbYImqUhbJ/PqdJ2sGhEdicq\n"
        "eC1V2n25ghnrRF8h2chbAAAAImtoaWNrZXlAS2V2aW5zLU1hY0Jvb2stUHJvLTQubG9jYW\n"
        "wBAgM="
    )
    service = DatabaseService(ssh_key=bare_letters)
    pkey = service._resolve_ssh_pkey()
    assert pkey is not None
    assert pkey.get_name() == "ssh-ed25519"


def test_resolve_ssh_pkey_from_single_line_with_escaped_newlines():
    single_line_escaped = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW"
        "QyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWwAAAKiL2tqii9ra"
        "ogAAAAtzc2gtZWQyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWw"
        "AAAEB49w67OFQKpqCG3y3uXEepz1YBUxWoZM0iYebm4Me8bbYImqUhbJ/PqdJ2sGhEdicq"
        "eC1V2n25ghnrRF8h2chbAAAAImtoaWNrZXlAS2V2aW5zLU1hY0Jvb2stUHJvLTQubG9jYW"
        "wBAgM=\\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    service = DatabaseService(ssh_key=single_line_escaped)
    pkey = service._resolve_ssh_pkey()
    assert pkey is not None
    assert pkey.get_name() == "ssh-ed25519"


def test_resolve_ssh_pkey_from_single_line_with_spaces():
    single_line_spaces = (
        "-----BEGIN OPENSSH PRIVATE KEY----- "
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW"
        "QyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWwAAAKiL2tqii9ra"
        "ogAAAAtzc2gtZWQyNTUxOQAAACC2CJqlIWyfz6nSdrBoRHYnKngtVdp9uYIZ60RfIdnIWw"
        "AAAEB49w67OFQKpqCG3y3uXEepz1YBUxWoZM0iYebm4Me8bbYImqUhbJ/PqdJ2sGhEdicq"
        "eC1V2n25ghnrRF8h2chbAAAAImtoaWNrZXlAS2V2aW5zLU1hY0Jvb2stUHJvLTQubG9jYW"
        "wBAgM= "
        "-----END OPENSSH PRIVATE KEY-----"
    )
    service = DatabaseService(ssh_key=single_line_spaces)
    pkey = service._resolve_ssh_pkey()
    assert pkey is not None
    assert pkey.get_name() == "ssh-ed25519"



