query_periods = """
SELECT
    CAST(d AS DATE) AS c_date,
    ROW_NUMBER() OVER (ORDER BY d) AS c_rank
FROM generate_series(
    DATE '{date_from}' + INTERVAL 1 DAY, 
    DATE '{date_from}' + INTERVAL {window} DAY, 
    INTERVAL 1 DAY
) AS t(d)
"""

query_procurements = f"""
WITH date_range AS ({query_periods}),
periods AS (
  SELECT 
    p_id,
    s_id,
    c_date,
    c_rank
  FROM date_range
  CROSS JOIN products
  CROSS JOIN stores
)
select 
  p_id,
  s_id,
  c_rank,
  dp_mean as pred_units_sold,
  (pc_id is not null) as needs_order
from periods
left join demandpredictions on 
  dp_p_id = p_id and dp_s_id = s_id and dp_date = c_date
left join procurements on
  pc_p_id = p_id and pc_s_id = s_id and 
  (pc_active_from <= c_date and c_date < pc_active_upto)
ORDER BY p_id, s_id, c_rank
"""
query_current_stocks = """
SELECT
    sk_p_id as p_id, 
    sk_s_id as s_id, 
    0 as c_rank, 
    sk_units as ending_inventory,
FROM stocks
WHERE sk_date == DATE '{date_from}'
"""
