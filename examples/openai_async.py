# examples/openai_async_example.py

import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openinference.instrumentation.openai import OpenAIInstrumentor

from dimredtracer import Tracer

load_dotenv()

# Env vars expected:
#   OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
#   OTEL_SERVICE_NAME
#   DIMRED_TENANT_ID  (optional)

async def main():
    # Initialize DimredTracer and OpenInference instrumentation
    tracer = Tracer(OpenAIInstrumentor())

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Add attributes to whatever span is current (parent span at this moment)
    tracer.set_attribute("experiment", "demo-async")
    tracer.set_attribute("user.id", "user-123")

    # ChatCompletion call will be auto-instrumented by OpenInference
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello from DimredTracer!"}],
    )

    # Add attributes to the ChatCompletion span (still the current span)
    tracer.set_attribute("postprocessing.latency_ms", 12)
    tracer.set_attribute("llm.response.preview", response.choices[0].message.content[:50])

    print(response.choices[0].message.content)

    # Flush spans at the end
    tracer.force_flush()


if __name__ == "__main__":
    asyncio.run(main())
