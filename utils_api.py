"""Utilities for working with OpenAI GPT APIs.
"""

import random
import functools
import json
import logging
import os
from io import BytesIO
import time
from multiprocessing import shared_memory

import numpy as np
import requests

from concurrent.futures import ThreadPoolExecutor

import openai
from openai import error as openai_error


def openai_complete(
    prompts,
    max_length,
    temperature,
    num_sampling=1,
    best_of=1,
    internal_batch_size=None,
    internal_num_sampling=None,
    sleep_time=3.0, # This is because of the rate limit: 20.000000 / min
    stop_token=None,
    logit_bias=None,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    logprobs=None,
    top_p=1.0,
  ):
  """OpenAI API call.
  Args:
    prompts: list of prompts
    max_length: max length of the output
    temperature: temperature of the output
    num_sampling: number of sampling
    best_of: number of best of
    internal_batch_size: internal batch size
    internal_num_sampling: internal number of sampling
    sleep_time: sleep time to avoid rate limit
    stop_token: stop token
    logit_bias: logit bias
    presence_penalty: presence penalty
    frequency_penalty: frequency penalty
    logprobs: logprobs
    top_p: top p
  Returns:
    list of responses
  """
  if type(prompts) is str:
    prompts = [prompts]

  def openai_api_call(prompts, api_key, organization):
    time.sleep(sleep_time + random.random())
    all_response = []


    all_logprobs = []
    accumulated_sleep_time = sleep_time
    if len(prompts) > 0:
      create_fn = openai.Completion.create

      if logit_bias is not None:
        create_fn = functools.partial(create_fn, logit_bias=json.loads(logit_bias))

      if logprobs is not None:
        create_fn = functools.partial(create_fn, logprobs=logprobs)

      if internal_batch_size is None:
        responses, accumulated_sleep_time = call_openai_internal(
            create_fn, prompts, max_length, best_of, num_sampling, stop_token,
            temperature, presence_penalty, frequency_penalty, top_p, api_key, organization,
            accumulated_sleep_time, sleep_time
        )
        all_response = [_["text"] for _ in responses["choices"]]

        if logprobs is not None:
          all_logprobs = [_["logprobs"] for _ in responses["choices"]]
      else:
        for start_idx in range(0, len(prompts), internal_batch_size):
          sub_prompts = prompts[start_idx:start_idx + internal_batch_size]

          if internal_num_sampling is None:
            responses, accumulated_sleep_time = call_openai_internal(
                create_fn, sub_prompts, max_length, best_of, num_sampling, stop_token,
                temperature, presence_penalty, frequency_penalty, top_p, api_key, organization,
                accumulated_sleep_time, sleep_time
            )
            if start_idx < len(prompts) - internal_batch_size:
              time.sleep(accumulated_sleep_time + random.random())
            all_response.extend([_["text"] for _ in responses["choices"]])
            if logprobs is not None:
              all_logprobs.extend([_["logprobs"] for _ in responses["choices"]])

          else:
            assert num_sampling == best_of
            assert num_sampling % internal_num_sampling == 0

            responses = dict()
            responses["choices"] = []
            stacked_responses = []
            for i in range(num_sampling // internal_num_sampling):
              response_choices, accumulated_sleep_time = call_openai_internal(
                  create_fn, sub_prompts, max_length, internal_num_sampling, internal_num_sampling, stop_token,
                  temperature, presence_penalty, frequency_penalty, top_p, api_key, organization,
                  accumulated_sleep_time, sleep_time
              )
              stacked_responses.append(response_choices["choices"])
              if start_idx < len(prompts) - internal_batch_size or i < num_sampling // internal_num_sampling - 1:
                time.sleep(accumulated_sleep_time + random.random())

            for i in range(len(stacked_responses[0])):
              for j in range(len(stacked_responses)):
                responses["choices"].append(stacked_responses[j][i])

            all_response.extend([_["text"] for _ in responses["choices"]])
            if logprobs is not None:
              all_logprobs.extend([_["logprobs"] for _ in responses["choices"]])
      return all_response, all_logprobs
    else:
      return None

  api_dicts = []
  multiple_api_key_file = "scripts/openai_keys.json"
  if os.path.exists(multiple_api_key_file):
    with open(multiple_api_key_file, "r") as f:
      lines = f.readlines()
      lines = "".join([_.strip() for _ in lines])
      lines = lines.replace("}{", "}[split]{")
      lines = lines.split("[split]")
      for line in lines:
        api_dicts.append(json.loads(line))

  if len(api_dicts) == 0:
    api_dicts = [{"api_key": openai.api_key, "organization": openai.organization}]

  targets = []
  targets_logprobs = []

  logging.info("Using %d API keys" % len(api_dicts))
  with ThreadPoolExecutor(max_workers=len(api_dicts)) as executor:
    futures = []
    for batch_idx, api_dict in enumerate(api_dicts):
      single_process_batch_size = ((len(prompts) - 1) // len(api_dicts)) + 1
      start_idx = single_process_batch_size * batch_idx
      end_idx = single_process_batch_size * (batch_idx + 1)

      if batch_idx == len(api_dicts) - 1:
        single_process_prompts = prompts[start_idx:]
      else:
        single_process_prompts = prompts[start_idx:end_idx]

      futures.append(
          executor.submit(
              openai_api_call,
              single_process_prompts,
              api_dict["api_key"],
              api_dict["organization"],
          ))

    for future in futures:
      responses = future.result()
      if responses is not None:
        targets.extend(responses[0])
        targets_logprobs.extend(responses[1])

  if len(targets_logprobs) > 0:
    return targets, targets_logprobs
  else:
    return targets



def call_openai_internal(create_fn, prompts, max_length, best_of, num_sampling, stop_token,
                         temperature, presence_penalty, frequency_penalty, top_p, api_key, organization,
                         accumulated_sleep_time, sleep_time):
  """Call OpenAI API with retry."""
  responses = None
  while responses is None:
    try:
      responses = create_fn(
          model="code-davinci-002",
          prompt=prompts,
          max_tokens=max_length,
          best_of=best_of,
          stop=stop_token,
          temperature=temperature,
          n=num_sampling,
          api_key=api_key,
          organization=organization,
          presence_penalty=presence_penalty,
          frequency_penalty=frequency_penalty,
          top_p=top_p,
      )
    except openai.error.RateLimitError as e:
      print(e)
      print(f"Batch size: {len(prompts)}, best_of: {best_of}, max_tokens: {max_length}")
      time.sleep(accumulated_sleep_time)
      accumulated_sleep_time += sleep_time
    except openai.error.APIError as e:
      print(e)
      print(f"Batch size: {len(prompts)}, best_of: {best_of}, max_tokens: {max_length}")
      print("API-Key:", api_key, "Organization:", organization)
      time.sleep(accumulated_sleep_time)
      accumulated_sleep_time += sleep_time
    except openai_error.Timeout as e:
      print(e)
      print("API-Key:", api_key, "Organization:", organization)
      time.sleep(accumulated_sleep_time)
      accumulated_sleep_time += sleep_time
    except openai_error.APIConnectionError as e:
      print(e)
      print("API-Key:", api_key, "Organization:", organization)
      time.sleep(accumulated_sleep_time)
      accumulated_sleep_time += sleep_time
  return responses, sleep_time


def blip_complete(
    images,
    texts,
    blip_urls,
    max_length=10,
    temperature=1.0,
    num_beams=5,
    length_penalty=-1.0,
    internal_batch_size=None,
):
  """BLIP API call.
  Args:
    images: list of images, as numpy arrays
    texts: list of texts
    blip_urls: list of blip api urls
    max_length: max length of the output
    temperature: temperature of the output
    num_beams: number of beams
    length_penalty: length penalty
    internal_batch_size: internal batch size
  Returns:
    list of responses
  """
  assert len(images) == len(texts)

  def blip_api_call(paired_image_text, url):
    response = None
    if len(paired_image_text) > 0:
      images = np.concatenate([img for img, _ in paired_image_text], axis=0)
      questions = [text for _, text in paired_image_text]

      port_number = url.split(":")[2].split("/")[0]
      NP_DATA_TYPE = np.float32
      MAX_BATCH_SIZE = 512
      NP_SHARED_NAME = f'npshared_{port_number}'
      shape_size = MAX_BATCH_SIZE * (224 * 224 * 3)
      d_size = np.dtype(NP_DATA_TYPE).itemsize * shape_size
      shm = shared_memory.SharedMemory(name=NP_SHARED_NAME, create=True, size=d_size)

      shared_images = np.ndarray((shape_size,), dtype=NP_DATA_TYPE, buffer=shm.buf)
      shared_images[:images.reshape(-1).shape[0]] = images.reshape(-1)
      shm.close()

      req = {
          "images_shape": images.shape,
          "texts": questions,
          "max_length": max_length,
          "temperature": temperature,
          "num_beams": num_beams,
          "length_penalty": length_penalty,
      }

      if internal_batch_size is not None:
        req["internal_batch_size"] = internal_batch_size

      res = requests.post(url, json=req)
      response = res.json()["targets"]
      shm.unlink()
    return response

  targets = []

  with ThreadPoolExecutor(max_workers=len(blip_urls)) as executor:
    futures = []
    for batch_idx, url in enumerate(blip_urls):
      single_process_batch_size = ((len(images) - 1) // len(blip_urls)) + 1
      start_idx = single_process_batch_size * batch_idx
      end_idx = single_process_batch_size * (batch_idx + 1)

      if batch_idx == len(blip_urls) - 1:
        single_process_paired_image_text = list(zip(images[start_idx:], texts[start_idx:]))
      else:
        single_process_paired_image_text = list(zip(images[start_idx:end_idx], texts[start_idx:end_idx]))

      futures.append(
          executor.submit(
              blip_api_call,
              single_process_paired_image_text,
              url,
          ))

    for future in futures:
      response = future.result()
      if response is not None:
        targets.extend(response)

  return targets


def blip_completev2(
    images,
    texts,
    blip_urls,
    max_length=10,
    temperature=1.0,
    num_beams=5,
    length_penalty=-1.0,
    internal_batch_size=None,
    encoding_format="JPEG",
):
  """BLIP API call.
  Args:
    images: list of images, as numpy arrays
    texts: list of texts
    blip_urls: list of blip api urls
    max_length: max length of the output
    temperature: temperature of the output
    num_beams: number of beams
    length_penalty: length penalty
    internal_batch_size: internal batch size
    encoding_format: encoding format of the image
  Returns:
    list of responses
  """
  assert len(images) == len(texts)

  def blip_api_call(paired_image_text, url):
    response = None
    if len(paired_image_text) > 0:
      headers = {
          "User-Agent": "BLIP-2 HuggingFace Space",
      }

      prompts = [text for _, text in paired_image_text]

      data = {
          "prompts": "[split]".join(prompts),
          "temperature": temperature,
          "length_penalty": length_penalty,
          "num_beams": num_beams,
          "max_length": max_length,
      }

      if internal_batch_size is not None:
        data["internal_batch_size"] = internal_batch_size

      files = {}
      for idx, (image, _) in enumerate(paired_image_text):
        image = encode_image(image, encoding_format=encoding_format)
        files[f"image{idx}"] = image

      response = requests.post(url, data=data, files=files, headers=headers).json()
    return response

  targets = []

  with ThreadPoolExecutor(max_workers=len(blip_urls)) as executor:
    futures = []
    for batch_idx, url in enumerate(blip_urls):
      single_process_batch_size = ((len(images) - 1) // len(blip_urls)) + 1
      start_idx = single_process_batch_size * batch_idx
      end_idx = single_process_batch_size * (batch_idx + 1)

      if batch_idx == len(blip_urls) - 1:
        single_process_paired_image_text = list(zip(images[start_idx:], texts[start_idx:]))
      else:
        single_process_paired_image_text = list(zip(images[start_idx:end_idx], texts[start_idx:end_idx]))

      futures.append(
          executor.submit(
              blip_api_call,
              single_process_paired_image_text,
              url,
          ))

    for future in futures:
      response = future.result()
      if response is not None:
        targets.extend(response)

  return targets


def encode_image(image, encoding_format="JPEG"):
  buffered = BytesIO()
  image.save(buffered, format=encoding_format)
  buffered.seek(0)
  return buffered
