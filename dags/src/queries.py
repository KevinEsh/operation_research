query_snapshot_raw = """
with sales_snapshot as (
  select 
    sa_date as dt, 
    sa_p_id as pid, 
    sa_s_id as sid,
    es_e_id as eid,
    p_group as pg,
    not (pr_id is null) as iop,
    ln(sa_units_sold + 1) as lus
  from sales
  left join eventstores on
    sa_date = es_date and sa_s_id = es_s_id
  left join promotions on
    sa_date = pr_date and sa_p_id = pr_p_id and sa_s_id = pr_s_id
  left join products on
    sa_p_id = p_id
),
-- lag_sales as (
--   select 
--     dt + 1 as l1d_dt,
--     dt + 2 as l2d_dt,
--     product_id,
--     store_id,
--     units_sold
--   from sales_snapshot
-- ),
next_sales as (
  select
    pid,
    sid,
    iop,
    eid,
    lus,
    dt - 1 as n1d_dt,
    dt - 2 as n2d_dt,
    dt - 3 as n3d_dt,
    dt - 4 as n4d_dt,
    dt - 5 as n5d_dt,
    dt - 6 as n6d_dt,
    dt - 7 as n7d_dt
  from sales_snapshot as ss
),
sales_featured as (
  select 
    ss.dt as c_date,
    ss.pid as product_id,
    ss.sid as store_id,
    ss.lus as log_units_sold,
    ss.pg as product_group,
    n1d.eid as next_1d_event_id,
    n2d.eid as next_2d_event_id,
    n3d.eid as next_3d_event_id,
    n4d.eid as next_4d_event_id,
    n5d.eid as next_5d_event_id,
    n6d.eid as next_6d_event_id,
    n7d.eid as next_7d_event_id,
    coalesce(n1d.iop, false) as next_1d_is_on_promo,
    coalesce(n2d.iop, false) as next_2d_is_on_promo,
    coalesce(n3d.iop, false) as next_3d_is_on_promo,
    coalesce(n4d.iop, false) as next_4d_is_on_promo,
    coalesce(n5d.iop, false) as next_5d_is_on_promo,
    coalesce(n6d.iop, false) as next_6d_is_on_promo,
    coalesce(n7d.iop, false) as next_7d_is_on_promo,
    n1d.lus as h1_log_units_sold,
    n2d.lus as h2_log_units_sold, 
    n3d.lus as h3_log_units_sold, 
    n4d.lus as h4_log_units_sold,
    n5d.lus as h5_log_units_sold,
    n6d.lus as h6_log_units_sold, 
    n7d.lus as h7_log_units_sold
    -- l1d.units_sold,
    -- l2d.units_sold
  from sales_snapshot as ss
  left join next_sales as n1d on 
    n1d.n1d_dt = dt and n1d.pid = ss.pid and n1d.sid = ss.sid
  left join next_sales as n2d on 
    n2d.n2d_dt = dt and n2d.pid = ss.pid and n2d.sid = ss.sid
  left join next_sales as n3d on 
    n3d.n3d_dt = dt and n3d.pid = ss.pid and n3d.sid = ss.sid
  left join next_sales as n4d on 
    n4d.n4d_dt = dt and n4d.pid = ss.pid and n4d.sid = ss.sid
  left join next_sales as n5d on 
    n5d.n5d_dt = dt and n5d.pid = ss.pid and n5d.sid = ss.sid
  left join next_sales as n6d on 
    n6d.n6d_dt = dt and n6d.pid = ss.pid and n6d.sid = ss.sid
  left join next_sales as n7d on 
    n7d.n7d_dt = dt and n7d.pid = ss.pid and n7d.sid = ss.sid
  -- left join lag_sales as l1d on l1d_dt = dt and l1d.pid = ss.pid and l1d.sid = ss.sid
  -- left join lag_sales as l2d on l2d_dt = dt and l2d.pid = ss.pid and l2d.sid = ss.sid
)
select *
from sales_featured
where 1=1
  and c_date between '{date_from}' and '{date_upto}'
order by product_id, store_id, c_date
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

query_demand_evolution = """
WITH RECURSIVE daily_data AS (
    SELECT
        dp.dp_date,
        dp.dp_p_id,
        dp.dp_s_id,
        dp.dp_mean AS pred_units_sold,
        COALESCE(df.units_sent, 0) AS units_sent,
        ROW_NUMBER() OVER (PARTITION BY dp.dp_p_id, dp.dp_s_id ORDER BY dp.dp_date) as rn
    FROM demandpredictions dp
    LEFT JOIN (
        SELECT
            df_date,
            df_p_id,
            df_s_id,
            SUM(df_packages_sent * tl_package_size) AS units_sent
        FROM demandfulfillments
        LEFT JOIN transportlinks ON tl_p_id = df_p_id AND tl_s_id = df_s_id AND tl_w_id = df_w_id
        WHERE df_date BETWEEN DATE '{date_from}' + INTERVAL '1 day' AND DATE '{date_from}' + INTERVAL '7 day'
          -- AND df_s_id = 1
        GROUP BY 1, 2, 3
    ) df ON dp.dp_date = df.df_date AND dp.dp_p_id = df.df_p_id AND dp.dp_s_id = df.df_s_id
    WHERE dp.dp_date BETWEEN DATE '{date_from}' + INTERVAL '1 day' AND DATE '{date_from}' + INTERVAL '7 day'
      -- AND dp.dp_s_id = 1
),
stocks_evolution AS (
    -- Anchor member: initial stock for the first day
    SELECT
        d.dp_date,
        d.dp_p_id,
        d.dp_s_id,
        d.pred_units_sold,
        d.units_sent,
        s.sk_units::bigint AS init_stocks,
        LEAST(s.sk_units + d.units_sent, d.pred_units_sold) AS met_demand,
        d.pred_units_sold - LEAST(s.sk_units + d.units_sent, d.pred_units_sold) AS unmet_demand,
        d.rn
    FROM daily_data d
    JOIN stocks s ON s.sk_p_id = d.dp_p_id AND s.sk_s_id = d.dp_s_id AND s.sk_date = DATE '{date_from}'
    WHERE d.rn = 1

    UNION ALL

    -- Recursive member: calculate next day's stock
    SELECT
        d.dp_date,
        d.dp_p_id,
        d.dp_s_id,
        d.pred_units_sold,
        d.units_sent,
        (p.init_stocks + p.units_sent - p.met_demand) AS init_stocks,
        LEAST((p.init_stocks + p.units_sent - p.met_demand) + d.units_sent, d.pred_units_sold) AS met_demand,
        d.pred_units_sold - LEAST((p.init_stocks + p.units_sent - p.met_demand) + d.units_sent, d.pred_units_sold) AS unmet_demand,
        d.rn
    FROM daily_data d
    JOIN stocks_evolution p ON d.dp_p_id = p.dp_p_id AND d.dp_s_id = p.dp_s_id AND d.rn = p.rn + 1
)
SELECT
    se.dp_date AS c_date,
    p.p_name,
    s.s_name,
    se.pred_units_sold::integer,
    se.units_sent::integer,
    se.init_stocks::integer,
    se.met_demand::integer,
    se.unmet_demand::integer
FROM stocks_evolution se
LEFT JOIN products p ON p.p_id = se.dp_p_id
LEFT JOIN stores s ON s.s_id = se.dp_s_id
ORDER BY s.s_name, p.p_name, se.dp_date
"""

query_stores_locations = """
select s_name, s_longitude, s_latitude
from stores
"""

query_workshops_locations = """
select w_name, w_longitude, w_latitude, w_capacity
from workshops
"""

query_sales_stocks = """
select 
  sa_date as c_date, 
  p_name, 
  s_name, 
  sa_units_sold as units_sold,
  sa_units_sold + 10 as units
from sales
left join products on p_id = sa_p_id
left join stores on s_id = sa_s_id
where sa_date between date '{date_from}' - interval '14 day' and '{date_from}'
"""

if __name__ == "__main__":
    # Example usage
    date_from = "{date_from}"
    q = query_demand_evolution.format(date_from=date_from, window=7, store_id=1)
    print(q)
