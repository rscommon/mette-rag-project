# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) project built around Facebook posts from Mette Frederiksen (Danish Prime Minister). The project is in its early stages.

## Data

- **`mette_frederiksen_posts.csv`** — ~2,337 Facebook posts scraped from Mette Frederiksen's page
- Columns: `ccpost_id`, `ccpageid`, `total_interactions`, `date`, `country`, `profile`, `facebook_url`, `post_url`, `post_text`
- Posts are in Danish, spanning from at least October 2025 to February 2026
- The `post_text` column contains the main content for RAG indexing

## Language

The dataset is in Danish. When building prompts, retrieval queries, or UI text that interacts with this data, default to Danish unless otherwise specified. So this is a change. 
