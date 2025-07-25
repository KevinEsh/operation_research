query_periods_dev = """
SELECT
    CAST(d AS DATE) AS c_date,
    ROW_NUMBER() OVER (ORDER BY d)::integer AS c_rank
FROM generate_series(
    DATE '{date_from}' + INTERVAL 1 DAY, 
    DATE '{date_from}' + INTERVAL {window} DAY, 
    INTERVAL 1 DAY
) AS t(d)
"""

query_periods = """
SELECT
    d::date AS c_date,
    ROW_NUMBER() OVER (ORDER BY d)::integer AS c_rank
FROM generate_series(
    DATE '{date_from}' + INTERVAL '1 day',
    DATE '{date_from}' + INTERVAL '{window} day',
    INTERVAL '1 day'
) AS d
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
    sk_units as ending_inventory
FROM stocks
WHERE sk_date = '{date_from}'
ORDER BY p_id, s_id
"""

query_transportlinks = f"""
WITH date_range AS ({query_periods})
SELECT
	tl_p_id as p_id,
	tl_s_id as s_id,
	tl_w_id as w_id,
	c_rank,
	tl_package_cost as package_cost,
	tl_package_size  as package_size
FROM transportlinks
CROSS JOIN date_range
ORDER BY 1,2,3,4
"""

query_workshops = f"""
WITH date_range AS ({query_periods})
SELECT
	w_id,
	c_rank,
	w_capacity as capacity
FROM workshops
CROSS JOIN date_range
"""
