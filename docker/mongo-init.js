// MongoDB initialization script for arxiv-corpus
// This runs when the container is first created

db = db.getSiblingDB('arxiv_corpus');

// Create collections with schema validation
db.createCollection('papers', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['arxiv_id', 'title', 'created_at'],
            properties: {
                arxiv_id: {
                    bsonType: 'string',
                    description: 'arXiv paper ID (e.g., "2301.12345")'
                },
                title: {
                    bsonType: 'string',
                    description: 'Paper title'
                },
                authors: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            name: { bsonType: 'string' },
                            affiliation: { bsonType: 'string' }
                        }
                    }
                },
                abstract: {
                    bsonType: 'string',
                    description: 'Paper abstract'
                },
                categories: {
                    bsonType: 'array',
                    items: { bsonType: 'string' },
                    description: 'arXiv categories'
                },
                published_date: {
                    bsonType: 'date',
                    description: 'Publication date'
                },
                updated_date: {
                    bsonType: 'date',
                    description: 'Last update date'
                },
                pdf_url: {
                    bsonType: 'string',
                    description: 'URL to PDF'
                },
                pdf_path: {
                    bsonType: 'string',
                    description: 'Local path to downloaded PDF'
                },
                text_path: {
                    bsonType: 'string',
                    description: 'Local path to extracted text'
                },
                status: {
                    enum: ['discovered', 'downloaded', 'extracted', 'processed', 'error'],
                    description: 'Processing status'
                },
                search_queries: {
                    bsonType: 'array',
                    items: { bsonType: 'string' },
                    description: 'Search queries that found this paper'
                },
                occurrence_count: {
                    bsonType: 'int',
                    description: 'Number of search queries that found this paper'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Record creation timestamp'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Record update timestamp'
                }
            }
        }
    }
});

db.createCollection('paragraphs', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['paper_id', 'text', 'created_at'],
            properties: {
                paper_id: {
                    bsonType: 'objectId',
                    description: 'Reference to papers collection'
                },
                arxiv_id: {
                    bsonType: 'string',
                    description: 'arXiv paper ID for quick lookup'
                },
                paragraph_index: {
                    bsonType: 'int',
                    description: 'Position in the document'
                },
                text: {
                    bsonType: 'string',
                    description: 'Paragraph text content'
                },
                tokens: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            text: { bsonType: 'string' },
                            lemma: { bsonType: 'string' },
                            pos: { bsonType: 'string' },
                            is_stop: { bsonType: 'bool' }
                        }
                    },
                    description: 'Tokenized content with linguistic annotations'
                },
                sentence_count: {
                    bsonType: 'int',
                    description: 'Number of sentences'
                },
                word_count: {
                    bsonType: 'int',
                    description: 'Number of words'
                },
                hits: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            term: { bsonType: 'string' },
                            list_name: { bsonType: 'string' },
                            rank: { bsonType: 'int' },
                            count: { bsonType: 'int' }
                        }
                    },
                    description: 'Term matches found in this paragraph'
                },
                total_hits: {
                    bsonType: 'int',
                    description: 'Total number of term hits'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Record creation timestamp'
                }
            }
        }
    }
});

db.createCollection('search_results', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['query', 'executed_at'],
            properties: {
                query: {
                    bsonType: 'string',
                    description: 'The search query string'
                },
                base_term: {
                    bsonType: 'string',
                    description: 'Base term used'
                },
                attribute: {
                    bsonType: 'string',
                    description: 'Attribute term used'
                },
                domain: {
                    bsonType: 'string',
                    description: 'Domain term used'
                },
                total_results: {
                    bsonType: 'int',
                    description: 'Total results returned by arXiv'
                },
                paper_ids: {
                    bsonType: 'array',
                    items: { bsonType: 'string' },
                    description: 'arXiv IDs found'
                },
                executed_at: {
                    bsonType: 'date',
                    description: 'When the search was executed'
                }
            }
        }
    }
});

db.createCollection('term_lists', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['name', 'terms', 'created_at'],
            properties: {
                name: {
                    bsonType: 'string',
                    description: 'Term list name'
                },
                description: {
                    bsonType: 'string',
                    description: 'Description of the term list'
                },
                terms: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            term: { bsonType: 'string' },
                            rank: { bsonType: 'int' },
                            is_wildcard: { bsonType: 'bool' },
                            expanded_forms: {
                                bsonType: 'array',
                                items: { bsonType: 'string' }
                            }
                        }
                    },
                    description: 'Terms in this list'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Record creation timestamp'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Record update timestamp'
                }
            }
        }
    }
});

// Create indexes for common queries
db.papers.createIndex({ arxiv_id: 1 }, { unique: true });
db.papers.createIndex({ status: 1 });
db.papers.createIndex({ categories: 1 });
db.papers.createIndex({ published_date: -1 });
db.papers.createIndex({ occurrence_count: -1 });
db.papers.createIndex({ 'search_queries': 1 });

db.paragraphs.createIndex({ paper_id: 1 });
db.paragraphs.createIndex({ arxiv_id: 1 });
db.paragraphs.createIndex({ total_hits: -1 });
db.paragraphs.createIndex({ 'hits.term': 1 });
db.paragraphs.createIndex({ text: 'text' });  // Full-text search index

db.search_results.createIndex({ query: 1 }, { unique: true });
db.search_results.createIndex({ executed_at: -1 });

db.term_lists.createIndex({ name: 1 }, { unique: true });

print('MongoDB initialization complete for arxiv_corpus database');
