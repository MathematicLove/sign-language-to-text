import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://hub.docker.com/v2"

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
    return sorted(set(names))

def main():
    p = argparse.ArgumentParser(description="Delete every Docker Hub tag except the one to keep.")
    p.add_argument("--user", default=os.environ.get("DOCKERHUB_USERNAME"))
    p.add_argument("--repo", default=os.environ.get("DOCKERHUB_REPO", "sign-language-to-text"))
    p.add_argument("--keep", default="latest")
    p.add_argument("--token-file", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.user:
        p.error("pass --user or set DOCKERHUB_USERNAME")
    token = os.environ.get("DOCKERHUB_TOKEN", "")
    if args.token_file:
        token = open(args.token_file, encoding="utf-8").read().strip()
    if not token:
        token = getpass.getpass("Docker Hub token (needs delete scope): ")

    jwt = call(API + "/users/login", {"username": args.user, "password": token})["token"]
    tags = all_tags(args.user, args.repo, jwt)
    stale = [t for t in tags if t != args.keep]
    print("tags on %s/%s: %s" % (args.user, args.repo, ", ".join(tags) or "none"))
    if args.keep not in tags:
        print("%s is missing, refusing to prune" % args.keep)
        return 1
    if not stale:
        print("nothing to delete")
        return 0
    if args.dry_run:
        print("would delete: %s" % ", ".join(stale))
        return 0
    failed = 0
    for name in stale:
        url = "%s/repositories/%s/%s/tags/%s/" % (API, args.user, args.repo, name)
        try:
            call(url, token=jwt, method="DELETE")
            print("deleted %s" % name)
        except urllib.error.HTTPError as exc:
            failed += 1
            print("could not delete %s: HTTP %d" % (name, exc.code))
            if exc.code == 403:
                print("  the token is missing the delete scope")
                print("  create one at hub.docker.com/settings/personal-access-tokens")
                print("  with permissions Read, Write, Delete")
                break
    if failed:
        return 1
    print("kept %s, deleted %d tag(s)" % (args.keep, len(stale)))
    return 0

if __name__ == "__main__":
    sys.exit(main())