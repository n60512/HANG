WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM clothing_interaction10
	ORDER BY RAND() 
	LIMIT 4160
	) 
SELECT clothing_interaction10.rank, clothing_interaction10.ID, clothing_interaction10.reviewerID, clothing_interaction10.`asin`, 
clothing_interaction10.overall, clothing_interaction10_rm_sw.reviewText, clothing_interaction10.unixReviewTime
FROM clothing_interaction10, clothing_interaction10_rm_sw
WHERE clothing_interaction10.reviewerID IN ( 
	-- SELECT * FROM clothing_interaction10_usertrain 
	SELECT * FROM rand_train_set 
) 
AND clothing_interaction10.ID = clothing_interaction10_rm_sw.ID
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM clothing_interaction10
	ORDER BY RAND() 
	LIMIT 4160
	) 
, tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction10
	WHERE reviewerID NOT IN (
		-- SELECT * FROM clothing_interaction10_usertrain 
		SELECT * FROM rand_train_set 
		) 
	LIMIT 520
	) 
SELECT clothing_interaction10.rank, clothing_interaction10.ID, clothing_interaction10.reviewerID, clothing_interaction10.`asin`, 
clothing_interaction10.overall, clothing_interaction10_rm_sw.reviewText, clothing_interaction10.unixReviewTime
FROM clothing_interaction10, clothing_interaction10_rm_sw
WHERE clothing_interaction10.reviewerID IN (SELECT * FROM tmptable) 
AND clothing_interaction10.ID = clothing_interaction10_rm_sw.ID
ORDER BY reviewerID,rank ASC ;
;