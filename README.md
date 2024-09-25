# SieveSync

A quality, zero-shot lipsync pipeline built with MuseTalk, LivePortrait, and CodeFormer. You can run the latest, production-ready version of SieveSync [here](https://www.sievedata.com/functions/sieve/lipsync) both through the playground and the API.

To learn more about the pipeline and see some example outputs, check out our [blog post](https://www.sievedata.com/blog/sievesync-zero-shot-lipsync-api-developers).

![SieveSync Cover](https://storage.googleapis.com/sieve-public-data/sievesync/sievesync-cover.webp)

The entire pipeline is built and hosted on [Sieve](https://sievedata.com).

## Usage

This repository contains the code for the pipeline we deploy to Sieve. It's meant for the developer community to have a nice way to experiment with lipsyncing pipelines. The community may find our implementation of face alignment and the pipeline architecture useful.

If you'd like to run this without Sieve, you may choose to swap out the model dependencies found at the top of `main.py` with MuseTalk, LivePortrait, and CodeFormer implementations.

### Through the public API

To run the pipeline through Sieve's API, you can follow the instructions [here](https://www.sievedata.com/functions/sieve/lipsync/guide).

### Deploying your own version of SieveSync

#### Prerequisites
1. Install [`ffmpeg`](https://ffmpeg.org/download.html)
2. `pip install -r requirements.txt`

[Create a Sieve account](https://www.sievedata.com/dashboard) and find your API key [here](https://www.sievedata.com/dashboard/settings). Then run the following command and enter your API key when prompted:

```bash
sieve login
```

Once you're logged in, you can deploy your own version of SieveSync with the following command:

```bash
cd sievesync
sieve deploy
```

You should then see a URL you can use to access your deployed function via the playground or API. As you make changes to the code, you can redeploy with the same command and you'll have an easy-to-use playground and production-ready API to test your changes.

You can also run this deployed function or Sieve's public function in a few lines of Python code:

```python
import sieve

# Run Sieve's public function
# You can change the slug to your deployed function's name
result = sieve.function.get("sieve/lipsync").run(
    file=sieve.File("elon-main.mp4"),
    audio=sieve.File("elon-spanish.wav"),
)

print("Output video: ", result.path)
```
