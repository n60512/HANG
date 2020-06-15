from utils.DBconnector import DBConnection
import tqdm

create_tmp_table = (
    """
    CREATE TEMPORARY TABLE tmp_table (
        `rank` BIGINT(21) NOT NULL,
        `ID` INT(11) NOT NULL DEFAULT '0',
        `reviewerID` CHAR(50) NULL DEFAULT NULL COLLATE 'utf8_general_ci',
        `asin` CHAR(50) NULL DEFAULT NULL COLLATE 'utf8_general_ci',
        `overall` FLOAT NULL DEFAULT NULL,
        `reviewText` TEXT NULL DEFAULT NULL COLLATE 'utf8_general_ci',
        `unixReviewTime` INT(11) NULL DEFAULT NULL
    );
    """
    )

def _select_candidate(conn):
    select_res = conn.selection('SELECT * FROM clothing_validation_reviewer_candidate;')
    reviewer = conn.selection('SELECT DISTINCT(reviewerID) FROM clothing_validation_reviewer_candidate;')


    for _row in select_res:

        _create_res = conn.execution(create_tmp_table)  # Create tmp table

        create_generated_sql = (
        """
        INSERT INTO tmp_table 
        SELECT * FROM clothing_interaction6_itembase
        WHERE `asin` = '#asin'
        AND rank < 7
        ORDER BY rank,unixReviewTime ASC ;
        """
        )

        print('rid:{}\tasin:{}'.format(
            _row['reviewerID'], _row['asin'])
            )

        _be_generate_asin = _row['asin']            # asin to be generated
        _be_generate_reviewer = _row['reviewerID']  # reviewer to be generated

        create_generated_sql = create_generated_sql.replace(
            '#asin', 
            _be_generate_asin
            )

        if(conn.execution(create_generated_sql)):  
            print('execution complete')

        _text = (
            """
            UPDATE tmp_table 
            SET reviewerID = '{}' 
            WHERE rank = 6;
            """
            ).format(_be_generate_reviewer)
        
        if(conn.execution(_text)):  
            print('updating complete')

        _show_res = False    
        if(_show_res):
            execute_res = conn.selection('SELECT * FROM tmp_table')
        
        _insert_into_table = (
            """
            INSERT INTO clothing_sparsity_generation_oringial_0527 
            SELECT * FROM tmp_table
            """
        )

        if(conn.execution(_insert_into_table)):
            print('Insertion complete')
        _drop = conn.execution('DROP TABLE tmp_table;')

        print(create_generated_sql)
        print('\n')
        stop = 1
        pass

    pass


if __name__ == "__main__":
    """
    創建用來生成第六筆評論的 Table (以User base建立)
    """


    conn = DBConnection()
    _select_candidate(conn)

    conn.close()


    pass
    

