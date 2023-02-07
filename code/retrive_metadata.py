import requests
import time
import json
import jsonlines
import argparse
import tqdm

import utils.keychain

bearer_token = utils.keychain.BEARER_TOKEN


# Connection
def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r

def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


# URLs
def create_url_like_lookup(id):
    url = f"https://api.twitter.com/2/tweets/{id}/liking_users"
    return url

def create_url_retweet_lookup(id):
    url = f"https://api.twitter.com/2/tweets/{id}/retweeted_by"
    return url

def create_url_author_lookup(id):
    tweet_fields = "tweet.fields=author_id"
    url = f"https://api.twitter.com/2/tweets?ids={id}&{tweet_fields}"
    return url

def create_url_author_metadata_lookup(user_id):
    url =  f"https://api.twitter.com/1.1/users/lookup.json?user_id={user_id}"
    return url


def retrieve_metadata(args):

    # Get a list of Tweet ids
    data_path = f'../data/labeled/CT23_1A_checkworthy_multimodal_english_{args.split}.jsonl'

    ids = []
    with jsonlines.open(data_path) as f:
        for obj in f:
            ids.append(obj['tweet_id'])
            
    # Also get a list from the ids already with retrived metadata
    retrieved_path = f'../data/retrieved/{args.split}.json'
    retrieved = {}
    try: 
        with open(retrieved_path, 'r') as f:
            retrieved = json.load(f)
    except:
        print('Retrieved json not found')
        pass

    out = {}
    for i, id in tqdm.tqdm(enumerate(ids)):
        if id not in retrieved.keys():
            try:
                out[id] = get_metadata(id)
            except Exception as e:
                print(e)
                break
    
    # Take the union of the two dicts
    for key, val in out.items():
        retrieved[key] = val
    
    # Save
    with open(retrieved_path, 'w') as f:
        json.dump(retrieved, f)

    return

def get_metadata(id):
    # # Number of likes
    url = create_url_like_lookup(id)
    json_response = connect_to_endpoint(url)

    # If the Tweet is not found
    if ('errors' in json_response.keys() and json_response['errors'][0]['title'] in ['Not Found Error', 'Authorization Error']):
        return {}
    elif 'meta' in json_response.keys() and 'result_count' in json_response['meta'].keys() and json_response['meta']['result_count'] == 0:
        n_likes = 0
    else:
        n_likes = len(json_response['data'])

    # Number of retweets
    url = create_url_retweet_lookup(id)
    json_response = connect_to_endpoint(url)
    if ('errors' in json_response.keys() and json_response['errors'][0]['title'] in ['Not Found Error', 'Authorization Error']):
        return {}
    elif 'meta' in json_response.keys() and json_response['meta'] == {'result_count': 0}:
        n_retweets = 0
    else:
        n_retweets = len(json_response['data'])

    # In the case where the Tweet is no longer available
    try:
        # Author
        url = create_url_author_lookup(id)
        json_response = connect_to_endpoint(url)
        author_id = json_response['data'][0]['author_id']

        # Author metadata: Name, verification status, #followed, #followers, #tweets, bio
        url = create_url_author_metadata_lookup(author_id)
        json_response = connect_to_endpoint(url)

        author_name = json_response[0]['name']
        bio = json_response[0]['description']
        n_followers = json_response[0]['followers_count']
        n_following = json_response[0]['following']
        verified = json_response[0]['verified']
        n_listed = json_response[0]['listed_count']

        return {
            'n_likes': n_likes,
            'n_retweets': n_retweets,
            'author_name': author_name,
            'bio': bio,
            'n_followers': n_followers,
            'n_following': n_following,
            'verified': verified,
            'n_listed': n_listed,
        }
    
    except:
        return {
            'n_likes': n_likes,
            'n_retweets': n_retweets,
        }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split', default='debug')

    args = parser.parse_args()

    for i in range(30):
        print(i)
        retrieve_metadata(args)
        time.sleep(61 * 15)
    