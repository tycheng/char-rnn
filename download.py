#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import bs4
import requests
import argparse


def process_page_title(soup):
    title = soup.title.string
    if "(" in title and ")" in title:
        index = title.index("(")
        title = title[:index]
    return title


def process_page(page_url):
    """ process text page and return text content """
    resp = requests.get(page_url)
    soup = bs4.BeautifulSoup(resp.content.decode("gbk", "ignore"), 'html.parser')
    div = soup.find("div", id="content")
    while div.find("ul", id="contentdp"):
        div.find("ul", id="contentdp").decompose()
    while div.find("div", class_="divimage"):
        div.find("div", class_="divimage").decompose()
    return process_page_title(soup), div.text


def fetch_novel(url):
    """ entrance """
    resp = requests.get(url)
    soup = bs4.BeautifulSoup(resp.content.decode("gbk", "ignore"), 'html.parser')
    novel = process_page_title(soup)
    if not os.path.exists(novel):
        os.makedirs(novel)
    for td in soup.find_all("td", class_="ccss"):
        for anchor in td.find_all("a"):
            next_page = url + anchor.attrs['href']
            title, text = process_page(next_page)
            filepath = os.path.join(novel, title + ".txt")
            with open(filepath, "w") as f:
                print(text)
                print(text, file=f)


def parse_args():
    default_url = "http://www.wenku8.com/novel/1/1657/"
    parser = argparse.ArgumentParser(prog="light novel downloader")
    parser.add_argument("--url", type=str, default=default_url)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fetch_novel(args.url)
