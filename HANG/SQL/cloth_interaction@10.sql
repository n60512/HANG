SELECT *
FROM clothing_interaction10 
WHERE reviewerID IN (SELECT * FROM clothing_interaction10_usertrain) 
ORDER BY reviewerID,rank ASC 
-- LIMIT 10000
;
WITH tmptable AS (
	SELECT DISTINCT(reviewerID)
	FROM clothing_interaction10
	WHERE reviewerID NOT IN (SELECT * FROM clothing_interaction10_usertrain) 
	LIMIT 520
	) 
SELECT *
FROM clothing_interaction10 
WHERE reviewerID IN (SELECT * FROM tmptable) 
ORDER BY reviewerID,rank ASC ;
;