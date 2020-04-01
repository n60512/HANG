from utils.DBconnector import DBConnection
import pickle
import tqdm

def _load_can2item(fpath = 'HANG/data/review_sparsity/review_sparsity_can2item.pickle'):
    # reload a file to a variable
    with open('{}'.format(fpath),  'rb') as file:
        a_dict1 =pickle.load(file)

    return a_dict1

def _create_sparsity_pair_table():

    conn = DBConnection()
    can2item = _load_can2item()

    for _can, _items in can2item.items():
        for _item in _items:
            sql_cmd = (
                'INSERT INTO clothing_sparsity_pair_42 (`reviewerID`, `asin`) '+
                'VALUES (\'{}\',\'{}\');'
                .format(_can, _item)
                )
            
            print(sql_cmd)
            res = conn.Insertion(sql_cmd)

    conn.close()
    pass


def _create_evaltable_tobe_generate():

    conn = DBConnection()
    
    _pair = conn.selection('SELECT * FROM clothing_sparsity_pair_42;')

    for _row in tqdm.tqdm(_pair):
        
        res = conn.selection(
            """
            WITH entire_item_base AS ( 
                SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
                ) 
            SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
            clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime 
            FROM clothing_interaction6_itembase 
            WHERE clothing_interaction6_itembase.`asin` IN (SELECT * FROM entire_item_base) 
            AND clothing_interaction6_itembase.rank <= 5 
            AND clothing_interaction6_itembase.`asin` = '{}' 
            ORDER BY `asin`,rank ASC 
            ;
            """.format(_row['asin'])
        )
        stop =1

        if len(res) != 0:

            insert_cmd = (
                """
                INSERT INTO clothing_sparsity_trainset_42 

                WITH entire_item_base AS ( 
                    SELECT DISTINCT(`asin`) FROM clothing_interaction6_itembase 
                    ) 
                SELECT clothing_interaction6_itembase.rank, clothing_interaction6_itembase.ID, clothing_interaction6_itembase.reviewerID, clothing_interaction6_itembase.`asin`, 
                clothing_interaction6_itembase.overall, clothing_interaction6_itembase.reviewText, clothing_interaction6_itembase.unixReviewTime 
                FROM clothing_interaction6_itembase 
                WHERE clothing_interaction6_itembase.`asin` IN (SELECT * FROM entire_item_base) 
                AND clothing_interaction6_itembase.rank <= 5 
                AND clothing_interaction6_itembase.`asin` = '{}' 
                ORDER BY `asin`,rank ASC 
                ;                
                """.format(_row['asin'])
            )
            _ins_res = conn.Insertion(insert_cmd)
            if(not _ins_res):
                """fail to insert"""
                stop = 1
            else:
                insert_cmd = (
                    """
                    INSERT INTO clothing_sparsity_trainset_42 
                    (rank, ID, reviewerID, `asin`, overall) 

                    WITH rating_record AS( 
                        SELECT clothing_review.ID, clothing_sparsity_pair_42.reviewerID, clothing_sparsity_pair_42.`asin`, clothing_review.overall 
                        FROM clothing_sparsity_pair_42, clothing_review 
                        WHERE clothing_sparsity_pair_42.`asin` = '{}' 
                        AND clothing_sparsity_pair_42.`reviewerID` = '{}' 	
                        AND clothing_sparsity_pair_42.reviewerID = clothing_review.reviewerID 
                        AND clothing_sparsity_pair_42.`asin` = clothing_review.`asin` 
                    ) 
                    SELECT 6, ID, reviewerID, `asin`, overall FROM rating_record 
                    ;
                    """.format(_row['asin'], _row['reviewerID'])
                )
                _ins_res_r6 = conn.Insertion(insert_cmd)
                if(not _ins_res_r6):
                    """fail to insert"""
                    pass

                # print('{}, {} success.'.format(_row['asin'], _row['reviewerID']))
                pass

    conn.close()
    pass

if __name__ == "__main__":

    # _create_sparsity_pair_table()
    _create_evaltable_tobe_generate()

    pass
    

