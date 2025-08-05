-- Initialize database for Agentic Scent Analytics
-- This script sets up the production database schema

-- Create audit events table
CREATE TABLE IF NOT EXISTS audit_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id VARCHAR(100),
    agent_id VARCHAR(100),
    resource VARCHAR(200),
    action VARCHAR(100) NOT NULL,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    security_level VARCHAR(20) DEFAULT 'internal',
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    signature VARCHAR(256),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent_id ON audit_events(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_success ON audit_events(success);

-- Create sensor readings table for historical data
CREATE TABLE IF NOT EXISTS sensor_readings (
    reading_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sensor_id VARCHAR(100) NOT NULL,
    sensor_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    values JSONB NOT NULL,
    metadata JSONB,
    quality_score DECIMAL(3,2) DEFAULT 1.0,
    batch_id VARCHAR(100),
    production_line VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for sensor readings
CREATE INDEX IF NOT EXISTS idx_sensor_readings_timestamp ON sensor_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_sensor_id ON sensor_readings(sensor_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_batch_id ON sensor_readings(batch_id);
CREATE INDEX IF NOT EXISTS idx_sensor_readings_production_line ON sensor_readings(production_line);

-- Create quality assessments table
CREATE TABLE IF NOT EXISTS quality_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id VARCHAR(100) NOT NULL,
    agent_id VARCHAR(100) NOT NULL,
    overall_quality DECIMAL(3,2) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    recommendation TEXT NOT NULL,
    passed_checks JSONB,
    failed_checks JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100)
);

-- Create indexes for quality assessments
CREATE INDEX IF NOT EXISTS idx_quality_batch_id ON quality_assessments(batch_id);
CREATE INDEX IF NOT EXISTS idx_quality_agent_id ON quality_assessments(agent_id);
CREATE INDEX IF NOT EXISTS idx_quality_created_at ON quality_assessments(created_at);

-- Create fingerprint models table
CREATE TABLE IF NOT EXISTS fingerprint_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id VARCHAR(100) NOT NULL UNIQUE,
    embedding_dim INTEGER NOT NULL,
    reference_fingerprint JSONB NOT NULL,
    similarity_threshold DECIMAL(5,4) NOT NULL,
    training_samples INTEGER NOT NULL,
    model_data JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for fingerprint models
CREATE INDEX IF NOT EXISTS idx_fingerprint_product_id ON fingerprint_models(product_id);

-- Create system metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    metadata JSONB,
    site_id VARCHAR(100),
    production_line VARCHAR(100)
);

-- Create indexes for system metrics
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON system_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON system_metrics(metric_name);

-- Create users table for authentication
CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(256) NOT NULL,
    salt VARCHAR(64) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for users
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

-- Create sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT true
);

-- Create indexes for sessions
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_active ON user_sessions(is_active);

-- Insert default admin user (password: admin123)
-- In production, this should be changed immediately
INSERT INTO users (username, password_hash, salt, permissions) 
VALUES (
    'admin',
    'pbkdf2_sha256$260000$demo_salt$hashed_password_placeholder',
    'demo_salt',
    '["read", "write", "admin"]'
) ON CONFLICT (username) DO NOTHING;

-- Create function to cleanup expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updating timestamps
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_fingerprint_models_updated_at 
    BEFORE UPDATE ON fingerprint_models 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions to application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO agentic;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agentic;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO agentic;

-- Create partitioning for large tables (optional, for high-volume deployments)
-- Partition audit_events by month
-- CREATE TABLE audit_events_y2024m01 PARTITION OF audit_events
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

COMMENT ON TABLE audit_events IS 'Comprehensive audit trail for regulatory compliance';
COMMENT ON TABLE sensor_readings IS 'Historical sensor data for analytics and trending';
COMMENT ON TABLE quality_assessments IS 'Quality control decisions and assessments';
COMMENT ON TABLE fingerprint_models IS 'Scent fingerprint models for product identification';
COMMENT ON TABLE system_metrics IS 'System performance and operational metrics';
COMMENT ON TABLE users IS 'Application users and authentication data';
COMMENT ON TABLE user_sessions IS 'Active user sessions for security tracking';