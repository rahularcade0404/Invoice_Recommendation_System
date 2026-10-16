import asyncio
from contextlib import asynccontextmanager
from difflib import SequenceMatcher, get_close_matches
from functools import partial
import os
from pathlib import Path
from typing import List
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd

# Import recommendation engine logic
from recommendation_engine import (
    recommend_products_item_based,
    recommend_majority_bought_products,
    recommend_state_majority_products,
)
from database.batch_db_pipeline import batch_generate_and_save_to_db
from data_preprocessing import load_and_preprocess_data
from matrix_construction import build_user_item_matrix
from cosine_similarity import compute_similarity_matrices
from database.database import (
    MajorityReorderRecommendation,
    SavedRecommendation,
    StateRecommendation as StateRecommendationRecord,
    get_db,
    init_db,
)
from sqlalchemy import Float, cast, func
from sqlalchemy.orm import Session


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await load_pipeline_data()
    yield


# Initialize FastAPI App
app = FastAPI(
    title="Invoice Data  Recommendation System API",
    description="Asynchronous ASGI API providing Collaborative Filtering, Reorder Analytics, and Regional State Recommendations.",
    version="1.0.0",
    lifespan=lifespan,
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "API_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Data Cache
DATA_CACHE = {}


def resolve_contact_name(requested_name: str, available_names) -> str:
    """Resolve a contact by trimmed, case-insensitive name or return suggestions."""
    contacts = list(dict.fromkeys(str(name).strip() for name in available_names))
    contacts_by_normalized_name = {
        name.casefold(): name for name in contacts if name
    }
    normalized_request = requested_name.strip().casefold()
    matched_name = contacts_by_normalized_name.get(normalized_request)
    if matched_name:
        return matched_name

    requested_terms = normalized_request.split()

    def shared_term_count(normalized_name: str) -> int:
        available_terms = normalized_name.split()
        return sum(
            any(
                requested_term == available_term
                or (
                    len(requested_term) >= 4
                    and available_term.startswith(requested_term[:4])
                )
                or (
                    len(available_term) >= 4
                    and requested_term.startswith(available_term[:4])
                )
                for available_term in available_terms
            )
            for requested_term in requested_terms
        )

    ranked_suggestions = sorted(
        (
            name
            for name in contacts_by_normalized_name
            if shared_term_count(name) > 0
        ),
        key=lambda name: (
            -shared_term_count(name),
            -SequenceMatcher(None, normalized_request, name).ratio(),
        ),
    )
    fuzzy_suggestions = get_close_matches(
        normalized_request,
        list(contacts_by_normalized_name),
        n=5,
        cutoff=0.4,
    )
    suggested_names = list(dict.fromkeys(ranked_suggestions + fuzzy_suggestions))[:5]
    raise HTTPException(
        status_code=404,
        detail={
            "message": f"Contact '{requested_name}' was not found in the invoice data.",
            "suggestions": [
                contacts_by_normalized_name[name] for name in suggested_names
            ],
        },
    )

DATA_FILE = Path(
    os.getenv(
        "INVOICE_DATA_FILE",
        Path(__file__).resolve().parent.parent
        / "data"
        / "Invoice_wise_detail_august.xlsx",
    )
)


async def refresh_database_recommendations(raw_df):
    """Refresh persisted recommendations without blocking API startup."""
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            batch_generate_and_save_to_db,
            raw_df,
        )
    except Exception as exc:
        print(f"Database recommendation refresh failed: {exc}")


async def load_pipeline_data():
    """Asynchronous startup event to load datasets and pre-compute user-item similarity matrices into memory."""
    init_db()
    file_path = str(DATA_FILE)

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Invoice workbook not found: {DATA_FILE}. "
            "Set INVOICE_DATA_FILE to the workbook's full path."
        )

    # Run blocking I/O data loading in an executor thread to keep the event loop non-blocking
    loop = asyncio.get_running_loop()
    raw_df = await loop.run_in_executor(
        None, load_and_preprocess_data, file_path
    )

    for col in [
        "Contact Name",
        "Material",
        "Material Desc",
        "Part Group",
        "Customer State Name",
        "MD",
        "Month",
    ]:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype(str).str.strip()

    user_item_df = (
        raw_df.groupby(
            ["Contact Name", "Material"],
            as_index=False,
        )
        .agg(Quantity=("Quantity", "sum"))
    )

    user_item_matrix = build_user_item_matrix(user_item_df)
    (
        part_group_month_sim_df,
        part_group_state_sim_df,
        material_part_group_sim_df,
    ) = compute_similarity_matrices(raw_df, user_item_matrix)

    DATA_CACHE["raw_df"] = raw_df
    DATA_CACHE["user_item_matrix"] = user_item_matrix
    DATA_CACHE["part_group_month_sim_df"] = part_group_month_sim_df
    DATA_CACHE["part_group_state_sim_df"] = part_group_state_sim_df
    DATA_CACHE["material_part_group_sim_df"] = material_part_group_sim_df

    asyncio.create_task(refresh_database_recommendations(raw_df.copy()))


# Pydantic Response Schemas
class ProductRecommendation(BaseModel):
    part_group: str = Field(..., alias="Part Group")
    material_desc: str = Field(..., alias="Material Desc")
    material: str = Field(..., alias="Material")
    recommendation_score: str = Field(..., alias="Recommendation Score")


class PersonalRecommendation(BaseModel):
    contact_name: str = Field(..., alias="Contact Name")
    recommended_part_group: str = Field(..., alias="Recommended Part Group")
    material: str = Field(..., alias="Material")
    material_desc: str = Field(..., alias="Material Desc")
    previously_bought_quantity: int = Field(..., alias="Previously Bought Quantity")
    recommended_quantity: int = Field(..., alias="Recommended Quantity")
    recommendation_score: str = Field(..., alias="Recommendation Score")


class StateRecommendation(BaseModel):
    contact_name: str = Field(..., alias="Contact Name")
    customer_state_name: str = Field(..., alias="Customer State Name")
    part_group: str = Field(..., alias="Part Group")
    previously_bought_qty: int = Field(..., alias="Previously Bought Qty")
    recommended_quantity: int = Field(..., alias="Recommended Quantity")
    recommendation_score: str = Field(..., alias="Recommendation Score")


# API Endpoints
@app.get("/health")
async def health_check():
    required_cache_keys = (
        "raw_df",
        "user_item_matrix",
        "part_group_month_sim_df",
        "part_group_state_sim_df",
        "material_part_group_sim_df",
    )
    missing_keys = [key for key in required_cache_keys if key not in DATA_CACHE]

    if missing_keys:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "message": "Recommendation data is not ready.",
                "missing": missing_keys,
            },
        )

    return {
        "status": "healthy",
        "message": "Recommendation Engine API is ready.",
        "customers": len(DATA_CACHE["user_item_matrix"].index),
        "records": len(DATA_CACHE["raw_df"]),
    }


@app.get("/recommendations/collaborative/{customer_id}",
    response_model=List[ProductRecommendation],
)
async def get_collaborative_recommendations(
    customer_id: str,
    top_n: int = Query(3, ge=1, le=10),
    min_threshold: float = Query(0.10, ge=0.0, le=0.5),
):
    """Fetches AI Collaborative Filtering Cross-Sell recommendations."""
    if "user_item_matrix" not in DATA_CACHE:
        raise HTTPException(
            status_code=503, detail="Pipeline data not initialized."
        )

    resolved_customer_id = resolve_contact_name(
        customer_id, DATA_CACHE["user_item_matrix"].index
    )

    loop = asyncio.get_event_loop()
    recs = await loop.run_in_executor(
        None,
        recommend_products_item_based,
        resolved_customer_id,
        DATA_CACHE["user_item_matrix"],
        DATA_CACHE["material_part_group_sim_df"],
        top_n,
        min_threshold,
    )

    if recs.empty:
        return []

    # Add catalog details for each recommended material while preserving the
    # recommendation ranking calculated by the collaborative engine.
    catalog = (
        DATA_CACHE["raw_df"]
        .dropna(subset=["Material", "Part Group", "Material Desc"])
        .groupby(["Material", "Part Group", "Material Desc"], as_index=False)
        .agg(Catalog_Quantity=("Quantity", "sum"))
        .sort_values("Catalog_Quantity", ascending=False)
        .drop_duplicates(subset=["Material"])
        [["Material", "Part Group", "Material Desc"]]
    )
    recs = recs.merge(catalog, on="Material", how="left")
    recs = recs[["Part Group", "Material Desc", "Material", "Score"]]

    recs["Recommendation Score"] = recs["Score"].map(lambda s: f"{s:.2f}%")
    return recs.to_dict(orient="records")


@app.get("/recommendations/majority/{customer_id}",
    response_model=List[PersonalRecommendation],
)
async def get_majority_reorders(customer_id: str, top_n: int = Query(3, ge=1, le=10)):
    """Fetches personal majority-bought product reorder recommendations."""
    if "raw_df" not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Pipeline data not initialized.")

    resolved_customer_id = resolve_contact_name(
        customer_id, DATA_CACHE["raw_df"]["Contact Name"].dropna().unique()
    )

    loop = asyncio.get_event_loop()
    recs = await loop.run_in_executor(
        None,
        recommend_majority_bought_products,
        resolved_customer_id,
        DATA_CACHE["raw_df"],
        top_n,
    )

    if recs.empty:
        return []

    recs["Recommendation Score"] = recs["Recommendation Score"].map(lambda s: f"{s:.2f}%")
    return recs.to_dict(orient="records")


@app.get("/recommendations/state/{customer_id}",
    response_model=List[StateRecommendation],)
async def get_state_recommendations(
    customer_id: str,
    top_n: int = Query(3, ge=1, le=10),
):
    """Fetches regional state-level majority recommendations."""
    if "raw_df" not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Pipeline data not initialized.")

    loop = asyncio.get_event_loop()
    recs = await loop.run_in_executor(
        None,
        partial(
            recommend_state_majority_products,
            customer_id,
            DATA_CACHE["raw_df"],
            top_n=top_n,
        ),
    )

    if recs.empty:
        customer_exists = (
            DATA_CACHE["raw_df"]["Contact Name"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(customer_id.strip().casefold())
            .any()
        )
        if not customer_exists:
            raise HTTPException(status_code=404,detail=f"Contact '{customer_id}' was not found in the invoice data.",)
        return []

    recs = recs.rename(columns={"Quantity": "Previously Bought Qty",})
    recs["Recommendation Score"] = recs["Recommendation Score"].map(lambda score: f"{score:.2f}%")

    return recs.to_dict(orient="records")

@app.get("/db/recommendations/{contact_name}")
async def get_saved_recommendations_from_db(
    contact_name: str, db: Session = Depends(get_db)
):
    """Fetch pre-computed recommendations stored for a contact."""
    normalized_contact = contact_name.strip().lower()
    collaborative_results = (
        db.query(SavedRecommendation)
        .filter(func.lower(SavedRecommendation.contact_name) == normalized_contact)
        .filter(SavedRecommendation.engine_type == "Collaborative")
        .order_by(SavedRecommendation.id)
        .all()
    )
    majority_results = (
        db.query(MajorityReorderRecommendation)
        .filter(func.lower(MajorityReorderRecommendation.contact_name) == normalized_contact)
        .order_by(MajorityReorderRecommendation.id)
        .all()
    )
    state_results = (
        db.query(StateRecommendationRecord)
        .filter(func.lower(StateRecommendationRecord.contact_name) == normalized_contact)
        .order_by(
            cast(
                func.replace(StateRecommendationRecord.recommendation_score, "%", ""),
                Float,
            ).desc()
        )
        .all()
    )

    if not collaborative_results and not majority_results and not state_results:
        raise HTTPException(status_code=404, detail="No stored recommendations found for contact.")

    results = [
        {
            "id": result.id,
            "contact_name": result.contact_name,
            "engine_type": result.engine_type,
            "recommended_part_group": result.recommended_part_group,
            "recommended_qty": result.recommended_qty,
            "recommendation_score": result.recommendation_score,
            "state": result.state,
            "generated_at": result.generated_at,
        }
        for result in collaborative_results
    ]
    results.extend(
        {
            "id": result.id,
            "contact_name": result.contact_name,
            "engine_type": "Majority_Reorder",
            "recommended_part_group": result.recommended_part_group,
            "material": result.material,
            "material_desc": result.material_desc,
            "previously_bought_qty": result.previously_bought_qty,
            "recommended_qty": result.recommended_qty,
            "recommendation_score": result.recommendation_score,
            "state": result.state,
            "generated_at": result.generated_at,
        }
        for result in majority_results
    )
    results.extend(
        {
            "id": result.id,
            "contact_name": result.contact_name,
            "engine_type": "State_Majority",
            "recommended_part_group": result.recommended_part_group,
            "previously_bought_qty": result.previously_bought_qty,
            "recommended_qty": result.recommended_qty,
            "recommendation_score": result.recommendation_score,
            "state": result.state,
            "generated_at": result.generated_at,
        }
        for result in state_results
    )
    return results


@app.get("/recommendations/state/{customer_id}/csv")
async def download_state_recommendations_csv(
    customer_id: str,
    top_n: int = Query(3, ge=1, le=10),
):
    """Fetch state recommendations as a downloadable CSV."""
    if "raw_df" not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Pipeline data not initialized.")

    # 1. Fetch recommendation data
    loop = asyncio.get_event_loop()
    recs = await loop.run_in_executor(
        None,
        partial(
            recommend_state_majority_products,
            customer_id,
            DATA_CACHE["raw_df"],
            top_n=top_n,
        ),
    )

    if recs.empty:
        raise HTTPException(
            status_code=404, detail="No recommendations found for this contact."
        )

    # 2. Format dataframe structure for tabular output
    item_col = (
        "Part Group" if "Part Group" in recs.columns else "Product ID"
    )
    formatted_df = recs.rename(
        columns={
            item_col: "Part Group/Product ID",
            "Customer_Bought_Qty": "Previously Bought Qty",
        }
    )

    if "Recommendation Score" in formatted_df.columns:
        scores = pd.to_numeric(
            formatted_df["Recommendation Score"], errors="coerce"
        ).clip(0, 100)
        formatted_df["Recommendation Score"] = scores.map(
            lambda score: f"{score:.2f}%" if pd.notna(score) else ""
        )

    previous_quantity = "Previously Bought Qty"
    if previous_quantity in formatted_df.columns:
        quantities = pd.to_numeric(formatted_df[previous_quantity], errors="coerce")
        if quantities.dropna().mod(1).eq(0).all():
            formatted_df[previous_quantity] = quantities.astype("Int64")

    # 3. Convert pandas DataFrame directly to a CSV string
    csv_content = formatted_df.to_csv(index=False)

    # 4. Stream response with CSV headers to trigger browser download
    clean_filename = customer_id.replace(" ", "_")
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=recommendations_{clean_filename}.csv"
        },
    )


@app.get("/recommendations/majority/{customer_id}/csv")
async def download_majority_reorders_csv(
    customer_id: str,
    top_n: int = Query(3, ge=1, le=10),
):
    """Fetch personal repeat-purchase recommendations as a downloadable CSV."""
    if "raw_df" not in DATA_CACHE:
        raise HTTPException(
            status_code=503, detail="Pipeline data not initialized."
        )

    loop = asyncio.get_event_loop()
    recs = await loop.run_in_executor(
        None,
        recommend_majority_bought_products,
        customer_id,
        DATA_CACHE["raw_df"],
        top_n,
    )

    recs["Recommendation Score"] = recs["Recommendation Score"].map(lambda score: f"{score:.2f}%")
    csv_content = recs.to_csv(index=False)
    clean_filename = customer_id.replace(" ", "_")

    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": (f"attachment; filename=repeat_purchases_{clean_filename}.csv")
        },
    )


# ASGI Runner Block
if __name__ == "__main__":
    uvicorn.run("api.api:app", host="127.0.0.1", port=8000, reload=True)