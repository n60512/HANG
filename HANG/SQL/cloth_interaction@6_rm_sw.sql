WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM clothing_interaction6
	ORDER BY RAND() 
	LIMIT 20000
	) 
SELECT clothing_interaction6.rank, clothing_interaction6.ID, clothing_interaction6.reviewerID, clothing_interaction6.`asin`, 
clothing_interaction6.overall, clothing_interaction6_rm_sw.reviewText, clothing_interaction6.unixReviewTime
FROM clothing_interaction6, clothing_interaction6_rm_sw
WHERE clothing_interaction6.reviewerID IN ( 
	-- SELECT * FROM clothing_interaction6_usertrain 
	SELECT * FROM rand_train_set 
) 
AND clothing_interaction6.ID = clothing_interaction6_rm_sw.ID
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM clothing_interaction6
	ORDER BY RAND() 
	LIMIT 20000
	) 
, tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction6
	WHERE reviewerID NOT IN (
		-- SELECT * FROM clothing_interaction6_usertrain 
		SELECT * FROM rand_train_set 
		) 
	LIMIT 2500
	) 
SELECT clothing_interaction6.rank, clothing_interaction6.ID, clothing_interaction6.reviewerID, clothing_interaction6.`asin`, 
clothing_interaction6.overall, clothing_interaction6_rm_sw.reviewText, clothing_interaction6.unixReviewTime
FROM clothing_interaction6, clothing_interaction6_rm_sw
WHERE clothing_interaction6.reviewerID IN (SELECT * FROM tmptable) 
AND clothing_interaction6.ID = clothing_interaction6_rm_sw.ID
ORDER BY reviewerID,rank ASC ;
;