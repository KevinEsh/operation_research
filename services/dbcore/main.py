from typing import List

from fastapi import Depends, FastAPI, HTTPException
from schema import (
    DemandPredictions,
    Events,
    EventStores,
    Procurements,
    ProductGroups,
    Products,
    Promotions,
    Sales,
    SQLModel,
    Stocks,
    Stores,
    TransportLinks,
    Workshops,
)
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, create_engine, select

app = FastAPI()

# check if the database file exists
# If not, it will be created when the first table is created
DB_PATH = "data/core.db"
DB_URI = f"duckdb:///{DB_PATH}"


def get_session():
    engine = create_engine(DB_URI)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


def get_error_msg(error):
    msg = str(error)
    if "Constraint Error:" in msg:
        return msg.split("Constraint Error:")[1].split("\n")[0].strip()
    return msg


@app.get("/productgroups")
def get_productgroups(session: Session = Depends(get_session)):
    """
    Get all product groups.
    """
    product_groups = session.exec(select(ProductGroups)).all()
    return {"productgroups": product_groups}


@app.post("/productgroups")
def bulk_create_productgroups(pgs: List[ProductGroups], session=Depends(get_session)):
    try:
        session.add_all(pgs)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))
    return {"productgroups": [pg.pg_id for pg in pgs]}


@app.patch("/productgroups/{pg_id}")
def update_productgroup(
    pg_id: int, pg: ProductGroups, session: Session = Depends(get_session)
):
    """Update a specific product group by ID."""
    existing_pg = session.get(ProductGroups, pg_id)
    if not existing_pg:
        raise HTTPException(status_code=404, detail="ProductGroup not found")

    pg_data = pg.model_dump(exclude_unset=True)
    existing_pg.sqlmodel_update(pg_data)

    try:
        session.add(existing_pg)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"message": "ProductGroup updated successfully", "productgroup_id": pg_id}


# @app.get("/productgroups/{pg_id}")
# def get_productgroup(pg_id: int, session: Session = Depends(get_session)):
#     """
#     Get a specific product group by ID.
#     """
#     pg = session.get(ProductGroups, pg_id)
#     if not pg:
#         raise HTTPException(status_code=404, detail="ProductGroup not found")
#     return pg


@app.get("/products")
def get_products(session: Session = Depends(get_session)):
    """
    Get all products.
    """
    products = session.exec(select(Products)).all()
    return {"products": products}


@app.post("/products")
def bulk_create_product(
    products: List[Products], session: Session = Depends(get_session)
):
    """
    Create products in bulk.
    This endpoint expects a list of Products.
    If no products are provided, it raises a 400 error.
    """
    if not products:
        raise HTTPException(status_code=400, detail="No products provided")

    try:
        session.add_all(products)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"products": [p.p_id for p in products]}


@app.put("/products/{p_id}")
def update_product(
    p_id: int, product: Products, session: Session = Depends(get_session)
):
    """Update a specific product by ID."""
    existing_product = session.get(Products, p_id)
    if not existing_product:
        raise HTTPException(status_code=404, detail="Product not found")

    for key, value in product.model_dump(exclude_unset=True).items():
        setattr(existing_product, key, value)

    try:
        session.add(existing_product)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"message": "Product updated successfully", "product_id": p_id}


# @app.get("/products/{p_id}")
# def get_product(p_id: int, session: Session = Depends(get_session)):
#     """
#     Get a specific product by ID.
#     """
#     product = session.get(Products, p_id)
#     if not product:
#         raise HTTPException(status_code=404, detail="Product not found")
#     return product


@app.get("/stores")
def get_stores(session: Session = Depends(get_session)):
    """
    Get all stores.
    """
    stores = session.exec(select(Stores)).all()
    return {"stores": stores}


@app.post("/stores")
def bulk_create_stores(stores: List[Stores], session: Session = Depends(get_session)):
    """
    Create stores in bulk.
    This endpoint expects a list of Stores.
    If no stores are provided, it raises a 400 error.
    """
    if not stores:
        raise HTTPException(status_code=400, detail="No stores provided")

    try:
        session.add_all(stores)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"stores": [s.s_id for s in stores]}


@app.get("/workshops")
def get_workshops(session: Session = Depends(get_session)):
    """
    Get all workshops.
    """
    workshops = session.exec(select(Workshops)).all()
    return {"workshops": workshops}


@app.post("/workshops")
def bulk_create_workshops(
    workshops: List[Workshops], session: Session = Depends(get_session)
):
    """
    Create workshops in bulk.
    This endpoint expects a list of Workshops.
    If no workshops are provided, it raises a 400 error.
    """
    if not workshops:
        raise HTTPException(status_code=400, detail="No workshops provided")

    try:
        session.add_all(workshops)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"workshops": [w.w_id for w in workshops]}


@app.get("/transportlinks")
def get_transportlinks(session: Session = Depends(get_session)):
    """
    Get all transport links.
    """
    transportlinks = session.exec(select(TransportLinks)).all()
    return {"transportlinks": transportlinks}


@app.post("/transportlinks")
def bulk_create_transportlinks(
    transportlinks: List[TransportLinks], session: Session = Depends(get_session)
):
    """
    Create transport links in bulk.
    This endpoint expects a list of TransportLinks.
    If no transport links are provided, it raises a 400 error.
    """
    if not transportlinks:
        raise HTTPException(status_code=400, detail="No transport links provided")

    try:
        session.add_all(transportlinks)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"transportlinks": [tl.tl_id for tl in transportlinks]}


@app.get("/procurements")
def get_procurements(session: Session = Depends(get_session)):
    """
    Get all procurements.
    """
    procurements = session.exec(select(Procurements)).all()
    return {"procurements": procurements}


@app.post("/procurements")
def bulk_create_procurements(
    procurements: List[Procurements], session: Session = Depends(get_session)
):
    """
    Create procurements in bulk.
    This endpoint expects a list of Procurements.
    If no procurements are provided, it raises a 400 error.
    """
    if not procurements:
        raise HTTPException(status_code=400, detail="No procurements provided")

    try:
        session.add_all(procurements)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"procurements": [pc.pc_id for pc in procurements]}


@app.get("/demandpredictions")
def get_demandpredictions(session: Session = Depends(get_session)):
    """
    Get all demand predictions.
    """
    demandpredictions = session.exec(select(DemandPredictions)).all()
    return {"demandpredictions": demandpredictions}


@app.post("/demandpredictions")
def bulk_create_demandpredictions(
    demandpredictions: List[DemandPredictions], session: Session = Depends(get_session)
):
    """
    Create demand predictions in bulk.
    This endpoint expects a list of DemandPredictions.
    If no demand predictions are provided, it raises a 400 error.
    """
    if not demandpredictions:
        raise HTTPException(status_code=400, detail="No demand predictions provided")

    try:
        session.add_all(demandpredictions)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"demandpredictions": [dp.dp_id for dp in demandpredictions]}


@app.get("/sales")
def get_sales(session: Session = Depends(get_session)):
    """
    Get all sales.
    """
    sales = session.exec(select(Sales)).all()
    return {"sales": sales}


@app.post("/sales")
def bulk_create_sales(sales: List[Sales], session: Session = Depends(get_session)):
    """
    Create sales in bulk.
    This endpoint expects a list of Sales.
    If no sales are provided, it raises a 400 error.
    """
    if not sales:
        raise HTTPException(status_code=400, detail="No sales provided")

    try:
        session.add_all(sales)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"sales": [sale.sa_id for sale in sales]}


@app.get("/promotions")
def get_promotions(session: Session = Depends(get_session)):
    """
    Get all promotions.
    """
    promotions = session.exec(select(Products)).all()
    return {"promotions": promotions}


@app.post("/promotions")
def bulk_create_promotions(
    promotions: List[Promotions], session: Session = Depends(get_session)
):
    """Create promotions in bulk.
    This endpoint expects a list of Products.
    If no promotions are provided, it raises a 400 error.
    """
    if not promotions:
        raise HTTPException(status_code=400, detail="No promotions provided")

    try:
        session.add_all(promotions)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"promotions": [pr.pr_id for pr in promotions]}


@app.get("/stocks")
def get_stocks(session: Session = Depends(get_session)):
    """
    Get all stocks.
    """
    stocks = session.exec(select(Stocks)).all()
    return {"stocks": stocks}


@app.post("/stocks")
def bulk_create_stocks(stocks: List[Stocks], session: Session = Depends(get_session)):
    """
    Create stocks in bulk.
    This endpoint expects a list of Stocks.
    If no stocks are provided, it raises a 400 error.
    """
    if not stocks:
        raise HTTPException(status_code=400, detail="No stocks provided")

    try:
        session.add_all(stocks)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"stocks": [stock.sk_id for stock in stocks]}


@app.get("/events")
def get_events(session: Session = Depends(get_session)):
    """
    Get all events.
    """
    events = session.exec(select(Events)).all()
    return {"events": events}


@app.post("/events")
def bulk_create_events(events: List[Events], session: Session = Depends(get_session)):
    """
    Create events in bulk.
    This endpoint expects a list of Events.
    If no events are provided, it raises a 400 error.
    """
    if not events:
        raise HTTPException(status_code=400, detail="No events provided")

    try:
        session.add_all(events)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"events": [event.e_id for event in events]}


@app.get("/eventstores")
def get_eventstores(session: Session = Depends(get_session)):
    """
    Get all event stores.
    """
    eventstores = session.exec(select(EventStores)).all()
    return {"eventstores": eventstores}


@app.post("/eventstores")
def bulk_create_eventstores(
    eventstores: List[EventStores], session: Session = Depends(get_session)
):
    """
    Create event stores in bulk.
    This endpoint expects a list of EventStores.
    If no event stores are provided, it raises a 400 error.
    """
    if not eventstores:
        raise HTTPException(status_code=400, detail="No event stores provided")

    try:
        session.add_all(eventstores)
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(status_code=400, detail=get_error_msg(error))

    return {"eventstores": [es.es_id for es in eventstores]}
