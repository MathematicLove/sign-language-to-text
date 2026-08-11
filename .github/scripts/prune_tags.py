import json
import os
import sys
import urllib.error
import urllib.request

API = "https://hub.docker.com/v2"
KEEP = "latest"


def call(url, data=None, token=None, method=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read().decode()
    return json.loads(raw) if raw else {}


def all_tags(user, repo, token):
    names = []
    url = "%s/repositories/%s/%s/tags/?page_size=100" % (API, user, repo)
    while url:
        page = call(url, token=token)
        names.extend(t["name"] for t in page.get("results", []))
        url = page.get("next")
    return names


def main():
    user = os.environ["DOCKERHUB_USERNAME"]
    token = os.environ["DOCKERHUB_TOKEN"]
    repo = os.environ.get("DOCKERHUB_REPO", "sign-language-to-text")

    jwt = call(API + "/users/login", {"username": user, "password": token})["token"]
    tags = all_tags(user, repo, jwt)
    stale = [t for t in tags if t != KEEP]
    print("tags on %s/%s: %s" % (user, repo, ", ".join(sorted(tags)) or "none"))
    if KEEP not in tags:
        print("%s is missing, refusing to prune" % KEEP)
        return 1
    for name in stale:
        url = "%s/repositories/%s/%s/tags/%s/" % (API, user, repo, name)
        try:
            call(url, token=jwt, method="DELETE")
            print("deleted %s" % name)
        except urllib.error.HTTPError as exc:
            print("could not delete %s: %s" % (name, exc.code))
            return 1
    print("kept %s, deleted %d tag(s)" % (KEEP, len(stale)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
