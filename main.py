from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

import core
from llm import llm

# no redirects, no docs routes: every route is handled explicitly in core.handle
app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def entry(request: Request, full_path: str):
    body = await request.body()
    status, headers, payload = await run_in_threadpool(
        core.handle, request.method, request.url.path, dict(request.headers),
        request.url.query, body, llm,
    )
    return Response(content=payload, status_code=status, headers=headers)
