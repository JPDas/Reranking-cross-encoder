from sentence_transformers import CrossEncoder
import streamlit as st

def get_result(query, passages, choice):
    if choice == "Nano":
        model_name = "model\ms-marco-TinyBERT-L-2"
    elif choice == "Small":
        model_name="model\ms-marco-MiniLM-L-12-v2"
    elif choice == "Medium":
        model_name="model\rank-T5-flan"
    elif choice == "Large":
        model_name="model\ms-marco-MultiBERT-L-12"

    ranker = CrossEncoder(model_name)
 
    model_inputs = [[query, passage["text"]] for passage in passages]
                    
    score =ranker.predict(model_inputs)

    tup = zip(passages, score)

    results = sorted(tup, key= lambda x : x[1], reverse=True)
    print(results)

    return results
 
st.set_page_config(
    layout="wide",
    page_title="ReRanking App"
)

def main():
    st.title("ReRanking using Sentence Transformers")
    st.sidebar.write("According to the Model Size 👇")
    menu = ["Nano", "Small", "Medium", "Large"]
    choice = st.sidebar.selectbox("Choose", menu)

    st.sidebar.info("""
**Model Options:**
- **Nano**: ~4MB, blazing fast model with competitive performance (ranking precision).
- **Small**: ~34MB, slightly slower with the best performance (ranking precision).
- **Medium**: ~110MB, slower model with the best zero-shot performance (ranking precision).
- **Large**: ~150MB, slower model with competitive performance (ranking precision) for 100+ languages.
""")

    with st.expander("About Cross Encoder"):
        st.markdown("""
        **Cross Encoder**: Ultra-lite & Super-fast Python library for search & retrieval re-ranking.

        - **Ultra-lite**: No heavy dependencies. Runs on CPU with a tiny ~4MB reranking model.
        - **Super-fast**: Speed depends on the number of tokens in passages and query, plus model depth.
        - **Cost-efficient**: Ideal for serverless deployments with low memory and time requirements.
        - **Based on State-of-the-Art Cross-encoders**: Includes models like ms-marco-TinyBERT-L-2-v2 (default), ms-marco-MiniLM-L-12-v2, rank-T5-flan, and ms-marco-MultiBERT-L-12.
        - **Sleek Models for Efficiency**: Designed for minimal overhead in user-facing scenarios.

        _Cross Encoder is tailored for scenarios requiring efficient and effective reranking, balancing performance with resource usage._
        """)

    query_input = "How to speedup LLMs?"
    context_input = [
        {
            "id":1,
            "text":"Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
            "meta": {"additional": "info1"}
        },
        {
            "id":2,
            "text":"LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
            "meta": {"additional": "info2"}
        },
        {
            "id":3,
            "text":"There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second. - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint. - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
            "meta": {"additional": "info3"}

        },
        {
            "id":4,
            "text":"Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
            "meta": {"additional": "info4"}
        },
        {
            "id":5,
            "text":"vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels",
            "meta": {"additional": "info5"}
        }
    ]
    submit_button = st.button('ReRank')

    if submit_button:
        with st.spinner('Processing...'):
            
            result = get_result(query_input, context_input, choice)
            st.subheader("Please find the ReRanked results below 👇")
            st.write(result)

if __name__ == "__main__":
    main()