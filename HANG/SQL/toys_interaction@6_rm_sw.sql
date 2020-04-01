WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM toys_interaction6
	ORDER BY RAND() 
	LIMIT 8000
	) 
SELECT toys_interaction6.rank, toys_interaction6.ID, toys_interaction6.reviewerID, toys_interaction6.`asin`, 
toys_interaction6.overall, toys_interaction6_rm_sw.reviewText, toys_interaction6.unixReviewTime
FROM toys_interaction6, toys_interaction6_rm_sw
WHERE toys_interaction6.reviewerID IN ( 
	-- SELECT * FROM toys_interaction6_usertrain 
	SELECT * FROM rand_train_set 
) 
AND toys_interaction6.ID = toys_interaction6_rm_sw.ID
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH rand_train_set AS (
	SELECT DISTINCT(reviewerID) FROM toys_interaction6
	ORDER BY RAND() 
	LIMIT 8000
	) 
, tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM toys_interaction6
	WHERE reviewerID NOT IN (
		-- SELECT * FROM toys_interaction6_usertrain 
		SELECT * FROM rand_train_set 
		) 
	LIMIT 1000
	) 
SELECT toys_interaction6.rank, toys_interaction6.ID, toys_interaction6.reviewerID, toys_interaction6.`asin`, 
toys_interaction6.overall, toys_interaction6_rm_sw.reviewText, toys_interaction6.unixReviewTime
FROM toys_interaction6, toys_interaction6_rm_sw
WHERE toys_interaction6.reviewerID IN (SELECT * FROM tmptable) 
AND toys_interaction6.ID = toys_interaction6_rm_sw.ID
ORDER BY reviewerID,rank ASC ;
;