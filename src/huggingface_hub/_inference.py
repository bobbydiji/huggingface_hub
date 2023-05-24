# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Related resources:
#    https://huggingface.co/tasks
#    https://huggingface.co/docs/huggingface.js/inference/README
#    https://github.com/huggingface/huggingface.js/tree/main/packages/inference/src
#    https://github.com/huggingface/text-generation-inference/tree/main/clients/python
#    https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/client.py
#    https://huggingface.slack.com/archives/C03E4DQ9LAJ/p1680169099087869
#    https://github.com/huggingface/unity-api#tasks
#
# Some TODO:
# - validate inputs/options/parameters? with Pydantic for instance? or only optionally?
# - add all tasks
# - handle async requests
#
# NOTE: the philosophy of this client is "let's make it as easy as possible to use it, even if less optimized". Some
# examples of how it translates:
# - Timeout / Server unavailable is handled by the client in a single "timeout" parameter.
# - Files can be provided as bytes, file paths, or URLs and the client will try to "guess" the type.
# - Images are parsed as PIL.Image for easier manipulation.
# - Provides a "recommended model" for each task => suboptimal but user-wise quicker to get a first script running.
# - Only the main parameters are publicly exposed. Power users can always read the docs for more options.

import base64
import io
import logging
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, ContextManager, Dict, Generator, List, Optional, Union, overload

from requests import HTTPError, Response

from ._inference_types import ClassificationOutput, ConversationalOutput, ImageSegmentationOutput
from .constants import INFERENCE_ENDPOINT
from .utils import build_hf_headers, experimental, get_session, hf_raise_for_status, is_pillow_available
from .utils._typing import Literal


if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


RECOMMENDED_MODELS = {
    "audio-classification": "superb/hubert-large-superb-er",
    "automatic-speech-recognition": "facebook/wav2vec2-large-960h-lv60-self",
    "conversational": "microsoft/DialoGPT-large",
    "image-classification": "google/vit-base-patch16-224",
    "image-segmentation": "facebook/detr-resnet-50-panoptic",
    "image-to-image": "timbrooks/instruct-pix2pix",
    "summarization": "facebook/bart-large-cnn",
    "text-to-speech": "espnet/kan-bayashi_ljspeech_vits",
}

UrlT = str
PathT = Union[str, Path]
BinaryT = Union[bytes, BinaryIO]
ContentT = Union[BinaryT, PathT, UrlT]


class InferenceTimeoutError(HTTPError, TimeoutError):
    """Error raised when a model is unavailable or the request times out."""


@experimental
class InferenceClient:
    def __init__(
        self, model: Optional[str] = None, token: Optional[str] = None, timeout: Optional[float] = None
    ) -> None:
        """
        Initialize a new Inference Client.

        [`InferenceClient`] aims to provide a unified experience to perform inference. The client can be used
        seamlessly with either the (free) Inference API or self-hosted Inference Endpoints.

        <Tip warning={true}>

        `InferenceClient` is still experimental. All tasks are not yet implemented and its API is subject to change in
        the future.

        </Tip>

        <Tip warning={true}>

        You must have `PIL` installed if you want to work with images (`pip install Pillow`).

        </Tip>

        Args:
            model (`str`, `optional`):
                The model to run inference with. Can be a model id hosted on the Hugging Face Hub, e.g.
                `bigcode/starcoder` or a URL to a deployed Inference Endpoint. Defaults to None, meaning the model can
                 be passed for each task or in last resort a recommended model will be used.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token.
            timeout (`float`, `optional`):
                The maximum number of seconds to wait for a response from the server. Loading a new model in Inference
                API can take up to several minutes. Defaults to None, meaning it will loop until the server is available.
        """
        self.model: Optional[str] = model
        self.headers = build_hf_headers(token=token)
        self.timeout = timeout

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    def post(
        self,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Response:
        """
        Make a POST request to the inference server.

        Args:
            json (Union[str, Dict, List], optional):
                The JSON data to send in the request body. Defaults to None.
            data (Union[str, Path, bytes, BinaryIO], optional):
                The content to send in the request body. It can be raw bytes, a pointer to an opened file, a local file
                path, or a URL to an online resource (image, audio file,...). If both `json` and `data` are passed,
                `data` will take precedence. At least `json` or `data` must be provided.. Defaults to None.
            model (str, optional):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. Will override the model defined at the instance level. Defaults to None.
            task (str, optional):
                The task to perform on the inference. Used only to default to a recommended model if `model` is not
                provided. At least `model` or `task` must be provided. Defaults to None.

        Returns:
            Response: The `requests` HTTP response.

        Raises:
            - [`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.
        """
        url = self._resolve_url(model, task)

        if data is not None and json is not None:
            warnings.warn("Ignoring `json` as `data` is passed as binary.")

        t0 = time.time()
        timeout = self.timeout
        while True:
            with _open_as_binary(data) as data_as_binary:
                try:
                    response = get_session().post(
                        url, json=json, data=data_as_binary, headers=self.headers, timeout=self.timeout
                    )
                except TimeoutError as error:
                    # Convert any `TimeoutError` to a `InferenceTimeoutError`
                    raise InferenceTimeoutError(f"Inference call timed out: {url}") from error

            try:
                hf_raise_for_status(response)
            except HTTPError as error:
                if error.response.status_code == 503:
                    # If Model is unavailable, either raise a TimeoutError...
                    if self.timeout is None or time.time() - t0 > self.timeout:
                        raise InferenceTimeoutError(
                            f"Model not loaded on the server: {url}. Please retry with a higher timeout."
                        ) from error
                    # ...or wait 1s and retry
                    logger.info(f"Waiting for model to be loaded on the server: {error}")
                    time.sleep(1)
                    if timeout is not None:
                        timeout = max(self.timeout - (time.time() - t0), 1)  # timeout of at least 1s
                    continue
                raise
            break
        return response

    def audio_classification(
        self,
        audio: ContentT,
        model: Optional[str] = None,
    ) -> ClassificationOutput:
        """
        Perform audio classification on the provided audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The audio content to classify. It can be raw audio bytes, a local audio file, or a URL pointing to an
                audio file.
            model (str, optional):
                The model to use for audio classification. Can be a model ID hosted on the Hugging Face Hub
                or a URL to a deployed Inference Endpoint. If not provided, the default recommended model for
                audio classification will be used.

        Returns:
            [`ClassificationOutput`]: The classification output containing the predicted label and its confidence.

        Raises:
            - [`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.

        Example:
            ```py
            >>> from huggingface_hub import InferenceClient
            >>> client = InferenceClient()
            >>> client.audio_classification(audio)
            [{'score': 0.4976358711719513, 'label': 'hap'}, {'score': 0.3677836060523987, 'label': 'neu'},...]
            ```
        """
        response = self.post(data=audio, model=model, task="audio-classification")
        return response.json()

    def automatic_speech_recognition(
        self,
        audio: ContentT,
        model: Optional[str] = None,
    ) -> str:
        """
        Perform automatic speech recognition (ASR or audio-to-text) on the given audio content.

        Args:
            audio (Union[str, Path, bytes, BinaryIO]):
                The content to transcribe. It can be raw audio bytes, local audio file, or a URL to an audio file.
            model (Optional[str], optional):
                The model to use for ASR. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. If not provided, the default recommended model for ASR will be used.

        Returns:
            str: The transcribed text.

        Raises:
            -[`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.

        Example:
            ```py
            >>> from huggingface_hub import InferenceClient
            >>> client = InferenceClient()
            >>> client.automatic_speech_recognition("hello_world.wav")
            "hello world"
            ```
        """
        response = self.post(data=audio, model=model, task="automatic-speech-recognition")
        return response.json()["text"]

    def conversational(
        self,
        text: str,
        generated_responses: Optional[List[str]] = None,
        past_user_inputs: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> ConversationalOutput:
        """
        Generate conversational responses based on the given input text (i.e. chat with the API).

        Args:
            text (str):
                The last input from the user in the conversation.
            generated_responses (List[str], optional):
                A list of strings corresponding to the earlier replies from the model. Defaults to None.
            past_user_inputs (Optional[List[str]], optional):
                A list of strings corresponding to the earlier replies from the user. Should be of the same length of
                `generated_responses`. Defaults to None.
            parameters (Dict[str, Any], optional):
                Additional parameters for the conversational task. Defaults to None. For more details about the available
                parameters, please refer to [this page](https://huggingface.co/docs/api-inference/detailed_parameters#conversational-task)
            model (Optional[str], optional):
                The model to use for the conversational task. Can be a model ID hosted on the Hugging Face Hub or a URL to
                a deployed Inference Endpoint. If not provided, the default recommended conversational model will be used.
                Defaults to None.

        Returns:
            [`ConversationalOutput`]: The generated conversational output.

        Raises:
            - [`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.

        Example:
            ```py
            >>> from huggingface_hub import InferenceClient
            >>> client = InferenceClient()
            >>> output = client.conversational("Hi, who are you?")
            >>> output
            {'generated_text': 'I am the one who knocks.', 'conversation': {'generated_responses': ['I am the one who knocks.'], 'past_user_inputs': ['Hi, who are you?']}, 'warnings': ['Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.']}
            >>> client.conversational(
            ...     "Wow, that's scary!",
            ...     generated_responses=output["conversation"]["generated_responses"],
            ...     past_user_inputs=output["conversation"]["past_user_inputs"],
            ... )
            ```
        """
        payload: Dict[str, Any] = {"inputs": {"text": text}}
        if generated_responses is not None:
            payload["inputs"]["generated_responses"] = generated_responses
        if past_user_inputs is not None:
            payload["inputs"]["past_user_inputs"] = past_user_inputs
        if parameters is not None:
            payload["parameters"] = parameters
        response = self.post(json=payload, model=model, task="conversational")
        return response.json()

    def image_classification(
        self,
        image: ContentT,
        model: Optional[str] = None,
    ) -> List[ClassificationOutput]:
        """
        Perform image classification on the given image using the specified model.

        Args:
            image (Union[str, Path, bytes, BinaryIO]):
                The image to classify. It can be raw bytes, an image file, or a URL to an online image.
            model (str, optional):
                The model to use for image classification. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image classification will be used.

        Returns:
            List[[`ClassificationOutput`]]: a list of dictionaries containing the predicted label and associated probability.

        Raises:
            - [`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.

        Example:
            ```py
            >>> from huggingface_hub import InferenceClient
            >>> client = InferenceClient()
            >>> client.image_classification("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg")
            [{'score': 0.9779096841812134, 'label': 'Blenheim spaniel'}, ...]
            ```
        """
        response = self.post(data=image, model=model, task="image-classification")
        return response.json()

    def image_segmentation(
        self,
        image: ContentT,
        model: Optional[str] = None,
    ) -> List[ImageSegmentationOutput]:
        """
        Perform image segmentation on the given image using the specified model.

        Args:
            image (Union[str, Path, bytes, BinaryIO]):
                The image to segment. It can be raw bytes, an image file, or a URL to an online image.
            model (str, optional):
                The model to use for image segmentation. Can be a model ID hosted on the Hugging Face Hub or a URL to a
                deployed Inference Endpoint. If not provided, the default recommended model for image segmentation will be used.

        Returns:
            List[`ImageSegmentationOutput`]: A list of dictionaries containing the segmented masks and associated attributes.

        Raises:
            - [`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.

        Example:
            ```py
            >>> from huggingface_hub import InferenceClient
            >>> client = InferenceClient()
            >>> client.image_segmentation("cat.jpg"):
            [{'score': 0.989008, 'label': 'LABEL_184', 'mask': <PIL.PngImagePlugin.PngImageFile image mode=L size=400x300 at 0x7FDD2B129CC0>}, ...]
            ```
        """

        # Segment
        response = self.post(data=image, model=model, task="image-segmentation")
        output = response.json()

        # Parse masks as PIL Image
        if not isinstance(output, list):
            raise ValueError(f"Server output must be a list. Got {type(output)}: {str(output)[:200]}...")
        for item in output:
            item["mask"] = _b64_to_image(item["mask"])
        return output

    def image_to_image(
        self,
        image: ContentT,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        **kwargs,
    ) -> "Image":
        """
        Perform image-to-image translation using a specified model.

        Args:
            image (Union[str, Path, bytes, BinaryIO]):
                The input image for translation. It can be raw bytes, an image file, or a URL to an online image..
            model (str, optional):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.
            prompt (str, optional):
                The text prompt to guide the image generation.
            negative_prompt (str, optional):
                A negative prompt to guide the translation process.
            height (int, optional):
                The height in pixels of the generated image.
            width (int, optional):
                The width in pixels of the generated image.
            num_inference_steps (int, optional):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (float, optional):
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.

        Returns:
            Image: The translated image.

        Raises:
            - [`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.

        Example:
            ```py
            >>> from huggingface_hub import InferenceClient
            >>> client = InferenceClient()
            >>> image = client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
            >>> image.save("tiger.jpg")
            ```
        """
        parameters = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }
        if all(parameter is None for parameter in parameters.values()):
            # Either only an image to send => send as raw bytes
            self.post(data=image, model=model, task="image-to-image")
            data = image
            payload: Optional[Dict[str, Any]] = None
        else:
            # Or an image + some parameters => use base64 encoding
            data = None
            payload = {"inputs": _b64_encode(image)}
            for key, value in parameters.items():
                if value is not None:
                    payload[key] = value

        response = self.post(json=payload, data=data, model=model, task="image-to-image")
        return _response_to_image(response)

    def summarization(
        self,
        text: str,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate a summary of a given text using a specified model.

        Args:
            text (str):
                The input text to summarize.
            parameters (Dict[str, Any], optional):
                Additional parameters for summarization. Check out this [page](https://huggingface.co/docs/api-inference/detailed_parameters#summarization-task)
                for more details.
            model (str, optional):
                The model to use for inference. Can be a model ID hosted on the Hugging Face Hub or a URL to a deployed
                Inference Endpoint. This parameter overrides the model defined at the instance level. Defaults to None.

        Returns:
            str: The generated summary text.

        Raises:
            - [`InferenceTimeoutError`]: If the model is unavailable or the request times out.
            - HTTPError: If the request fails with an HTTP error status code other than HTTP 503.

        Example:
            ```py
            >>> from huggingface_hub import InferenceClient
            >>> client = InferenceClient()
            >>> client.summarization("The Eiffel tower...")
            'The Eiffel tower is one of the most famous landmarks in the world....'
            ```
        """
        payload: Dict[str, Any] = {"inputs": text}
        if parameters is not None:
            payload["parameters"] = parameters
        response = self.post(json=payload, model=model, task="summarization")
        return response.json()[0]["summary_text"]

    def text_to_speech(self, text: str, model: Optional[str] = None) -> bytes:
        response = self.post(json={"inputs": text}, model=model, task="text-to-speech")
        return response.content

    def _resolve_url(self, model: Optional[str], task: Optional[str]) -> str:
        model = model or self.model

        # If model is already a URL, ignore `task` and return directly
        if model is not None and (model.startswith("http://") or model.startswith("https://")):
            return model

        # # If no model but task is set => fetch the recommended one for this task
        if model is None:
            if task is None:
                raise ValueError(
                    "You must specify at least a model (repo_id or URL) or a task, either when instantiating"
                    " `InferenceClient` or when making a request."
                )
            model = _get_recommended_model(task)

        # TODO: handle when task is feature-extraction / sentence-similarity
        #       i.e. the only case where a model has several useful tasks

        # Compute InferenceAPI url
        return f"{INFERENCE_ENDPOINT}/models/{model}"


def _get_recommended_model(task: str) -> str:
    # TODO: load from a config file? (from the Hub?) Would make sense to make updates easier.
    if task in RECOMMENDED_MODELS:
        model = RECOMMENDED_MODELS[task]
        logger.info(f"Defaulting to recommended model {model} for task {task}.")
        return model
    raise NotImplementedError()


@overload
def _open_as_binary(content: ContentT) -> ContextManager[BinaryT]:
    ...  # means "if input is not None, output is not None"


@overload
def _open_as_binary(content: Literal[None]) -> ContextManager[Literal[None]]:
    ...  # means "if input is None, output is None"


@contextmanager  # type: ignore
def _open_as_binary(content: Optional[ContentT]) -> Generator[Optional[BinaryT], None, None]:
    """Open `content` as a binary file, either from a URL, a local path, or raw bytes.

    Do nothing if `content` is None,

    TODO: handle a PIL.Image as input
    TODO: handle base64 as input
    """
    # If content is a string => must be either a URL or a path
    if isinstance(content, str):
        if content.startswith("https://") or content.startswith("http://"):
            logger.debug(f"Downloading content from {content}")
            yield get_session().get(content).content  # TODO: retrieve as stream and pipe to post request ?
            return
        content = Path(content)
        if not content.exists():
            raise FileNotFoundError(
                f"File not found at {content}. If `data` is a string, it must either be a URL or a path to a local"
                " file. To pass raw content, please encode it as bytes first."
            )

    # If content is a Path => open it
    if isinstance(content, Path):
        logger.debug(f"Opening content from {content}")
        with content.open("rb") as f:
            yield f
    else:
        # Otherwise: already a file-like object or None
        yield content


def _b64_encode(content: ContentT) -> str:
    """Encode a raw file (image, audio) into base64. Can be byes, an opened file, a path or a URL."""
    with _open_as_binary(content) as data:
        data_as_bytes = data if isinstance(data, bytes) else data.read()
        return base64.b64encode(data_as_bytes).decode()


def _b64_to_image(encoded_image: str) -> "Image":
    """Parse a base64-encoded string into a PIL Image."""
    Image = _import_pil_image()
    return Image.open(io.BytesIO(base64.b64decode(encoded_image)))


def _response_to_image(response: Response) -> "Image":
    """Parse a Response object into a PIL Image.

    Expects the response body to be raw bytes. To deal with b64 encoded images, use `_b64_to_image` instead.
    """
    Image = _import_pil_image()
    return Image.open(io.BytesIO(response.content))


def _import_pil_image():
    """Make sure `PIL` is installed on the machine."""
    if not is_pillow_available():
        raise ImportError(
            "Please install Pillow to use deal with images (`pip install Pillow`). If you don't want the image to be"
            " post-processed, use `client.post(...)` and get the raw response from the server."
        )
    from PIL import Image

    return Image


if __name__ == "__main__":
    client = InferenceClient()

    # Text to speech to text
    audio = client.text_to_speech("Hello world")
    client.audio_classification(audio)
    client.automatic_speech_recognition(audio)

    # Image classification
    client.image_classification("cat.jpg")
    client.image_classification(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
    )

    # Image segmentation
    for item in client.image_segmentation("cat.jpg"):
        item["mask"].save(f"cat_{item['label']}_{item['score']}.jpg")

    # Image to image (instruct pix2pix)
    image = client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
    image.save("tiger.jpg")

    # Text summary
    client.summarization("The Eiffel tower...")

    # Chat
    output = client.conversational("Hi, who are you?")
    client.conversational(
        "Wow, that's scary!",
        generated_responses=output["conversation"]["generated_responses"],
        past_user_inputs=output["conversation"]["past_user_inputs"],
    )
