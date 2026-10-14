//! Client defaults use the shared transport, independently of response codecs.
use std::error::Error as StdError;
use std::time::Duration;

use httpmock::MockServer;
use lithos_llm::Client;
use lithos_llm::catalog::Catalog;
use lithos_llm::credentials::{CredentialHeader, Credentials, SecretValue, StaticCredentials};
use lithos_llm::middleware::{RetryMiddleware, RetryPolicy};
use lithos_llm::types::ErrorKind;

use crate::support::{self, WireProvider};

#[tokio::test]
async fn scoped_defaults_are_isolated_across_clients_and_provider_routes()
-> Result<(), Box<dyn StdError>> {
    let server = MockServer::start_async().await;
    for codec in [
        "openai-chat",
        "openai-responses",
        "anthropic-messages",
        "gemini-generate",
    ] {
        let provider = WireProvider::new("custom", codec, "model");
        let base = support::client_for(
            provider.catalog(&server.base_url()),
            "custom",
            Credentials::none(),
        );
        let first = base.with_default_headers([("X-Session-Id", "first")])?;
        let second = first.with_default_headers([("x-session-ID", "second")])?;
        let request = support::base_request(&provider.selector());
        for (client, expected) in [
            (&base, None),
            (&first, Some("first")),
            (&second, Some("second")),
        ] {
            let mock = server
                .mock_async(|when, then| {
                    let when = when.method("POST");
                    if let Some(value) = expected {
                        when.header("x-session-id", value);
                    } else {
                        when.header_missing("x-session-id");
                    }
                    then.status(400)
                        .body("deliberate rejection after capturing headers");
                })
                .await;
            let (complete, stream) = tokio::join!(
                client.complete(request.clone()),
                client.stream(request.clone())
            );
            assert!(complete.is_err());
            assert!(stream.is_err());
            let expected_calls = if codec == "anthropic-messages" {
                assert!(client.count_input_tokens(request.clone()).await.is_err());
                3
            } else {
                2
            };
            mock.assert_calls_async(expected_calls).await;
            mock.delete_async().await;
        }
    }
    Ok(())
}

#[tokio::test]
async fn explicit_headers_win_and_scoped_defaults_survive_retries() -> Result<(), Box<dyn StdError>>
{
    let server = MockServer::start_async().await;
    let provider = WireProvider::new("custom", "openai-chat", "model");
    let catalog = Catalog::builder()
        .overlay_toml(&provider.toml(&server.base_url()))?
        .overlay_toml("[providers.custom]\ndefault_headers = { X-Session-ID = 'configured' }")?
        .build()?;
    let client = Client::builder()
        .catalog(catalog)
        .credentials(StaticCredentials::new().with(
            "custom",
            Credentials::headers([CredentialHeader::new(
                "x-credential",
                SecretValue::new("credential"),
            )]),
        ))
        .middleware(RetryMiddleware::new(
            RetryPolicy::exponential()
                .max_attempts(2)
                .initial_delay(Duration::ZERO),
        ))
        .build()?
        .client
        .with_default_headers([
            ("x-session-id", "default"),
            ("x-credential", "default"),
            ("x-trace", "trace"),
        ])?;
    let mock = server
        .mock_async(|when, then| {
            when.method("POST")
                .header("x-session-id", "configured")
                .header("x-credential", "credential")
                .header("x-trace", "trace");
            then.status(503).body("retry");
        })
        .await;
    assert!(
        client
            .complete(support::base_request(&provider.selector()))
            .await
            .is_err()
    );
    mock.assert_calls_async(2).await;
    Ok(())
}

#[test]
fn invalid_defaults_fail_before_dispatch() {
    let provider = WireProvider::new("custom", "openai-chat", "model");
    let client = support::client_for(
        provider.catalog("http://127.0.0.1"),
        "custom",
        Credentials::none(),
    );
    for header in [("bad name", "value"), ("x-trace", "value\r\ninjected: yes")] {
        let error = client
            .with_default_headers([header])
            .expect_err("invalid header");
        assert_eq!(error.kind(), ErrorKind::Configuration);
        assert!(!format!("{error:?}").contains("injected"));
    }
}
