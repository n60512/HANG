WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM clothing_interaction6_maxlen30
	ORDER BY RAND() 
	LIMIT 8000
	) 
SELECT clothing_interaction6_maxlen30.rank, clothing_interaction6_maxlen30.ID, clothing_interaction6_maxlen30.reviewerID, clothing_interaction6_maxlen30.`asin`, 
clothing_interaction6_maxlen30.overall, clothing_interaction6_maxlen30.reviewText, clothing_interaction6_maxlen30.unixReviewTime
FROM clothing_interaction6_maxlen30
WHERE clothing_interaction6_maxlen30.reviewerID IN ( 
	SELECT * FROM rand_train_set 
) 
-- AND clothing_interaction10.ID = clothing_interaction10_rm_sw.ID
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM clothing_interaction6_maxlen30
	ORDER BY RAND() 
	LIMIT 8000
	) 
, tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction6_maxlen30
	WHERE reviewerID NOT IN (
		-- SELECT * FROM clothing_interaction10_usertrain 
		SELECT * FROM rand_train_set 
		) 
	LIMIT 2000
	) 
SELECT clothing_interaction6_maxlen30.rank, clothing_interaction6_maxlen30.ID, clothing_interaction6_maxlen30.reviewerID, clothing_interaction6_maxlen30.`asin`, 
clothing_interaction6_maxlen30.overall, clothing_interaction6_maxlen30.reviewText, clothing_interaction6_maxlen30.unixReviewTime
FROM clothing_interaction6_maxlen30
WHERE clothing_interaction6_maxlen30.reviewerID IN (SELECT * FROM tmptable) 
-- AND clothing_interaction10.ID = clothing_interaction10_rm_sw.ID
ORDER BY reviewerID,rank ASC ;
;