-- Drop tables if exists
DROP TABLE IF EXISTS news_df;

-- Create clean_data_df table for raw dataset
CREATE TABLE news_df (
	id SERIAL PRIMARY KEY,
	topic VARCHAR,
	news_date DATE,
	label NUMERIC,
	combined_text VARCHAR
);