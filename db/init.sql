CREATE TABLE IF NOT EXISTS scored_transactions (
    transaction_id  TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    amount          DOUBLE PRECISION NOT NULL,
    merchant_category TEXT NOT NULL,
    fraud_score     DOUBLE PRECISION NOT NULL,
    is_fraud        BOOLEAN NOT NULL,
    shap_explanation JSONB NOT NULL,
    scored_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scored_transactions_scored_at ON scored_transactions (scored_at);
