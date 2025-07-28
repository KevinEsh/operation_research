SELECT 'CREATE DATABASE dbcore'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'dbcore')\gexec
SELECT 'CREATE DATABASE dbmlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'dbmlflow')\gexec
SELECT 'CREATE DATABASE dbmetaflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'dbmetaflow')\gexec
SELECT 'CREATE DATABASE dbairflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'dbairflow')\gexec