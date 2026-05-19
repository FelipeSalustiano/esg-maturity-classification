import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from urllib.parse import quote_plus
import logging
import os

logging.basicConfig(level=logging.INFO)

def silver_transformer(filepath: str):

    try:
        load_dotenv()

        USER = os.getenv("user")
        PASSWORD = quote_plus(os.getenv("password"))
        HOST = os.getenv("host")
        PORT = os.getenv("port")
        DATABASE = os.getenv("database")

        engine = create_engine(
            f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        )

        query = """
        SELECT
            "Hora de início" AS hora_inicio,
            "Hora de conclusão" AS hora_conclusao,
            "Hora da última modificação" AS ultima_modificacao,

            "Qual é a razão social e o nome fantasia da sua empresa?"
                AS empresa,

            "Qual é o CNPJ da empresa?"
                AS cnpj,

            "Quantos colaboradores a empresa possui atualmente?"
                AS qtd_colaboradores,

            "Qual é o faturamento médio anual da empresa?"
                AS faturamento_anual,

            "Qual cargo ou função atual na empresa?"
                AS cargo,

            "Existem processos internos para garantir o cumprimento da legislação ambiental e de segurança do trabalho?"
                AS possui_processos_legais,

            "Nos últimos cinco anos, a empresa recebeu alguma sanção administrativa de natureza ambiental e de segurança do trabalho?"
                AS recebeu_sancao,

            "A empresa possui compromisso formal com o trabalho digno, incluindo a não utilização de trabalho infantil, forçado ou análogo à escravidão, e o cumprimento integral da legislação trabalhista vigente?"
                AS trabalho_digno,

            "A empresa possui certificações relacionadas à gestão ambiental, segurança do trabalho ou conformidade legal (ex: ISO 14001, ISO 45001, OHSAS 18001, FSC, entre outras)?"
                AS possui_certificacoes,

            "A empresa possui uma política formal implementada relacionada à responsabilidade sociolambiental (como política ambiental, de sustentabilidade ou equivalente)?"
                AS possui_politica_esg,

            "A empresa realiza treinamentos sobre temas de sustentabilidade com seus colaboradores?"
                AS treinamento_sustentabilidade,

            "A empresa possui práticas formais para garantir o compliance do negócio, como políticas anticorrupção, cumprimento de normas legais e promoção da transparência?"
                AS possui_compliance,

            "A empresa realiza o levantamento de seus aspectos e impactos ambientais e estabelece controle para eles?"
                AS controla_impactos_ambientais,

            "A empresa possui práticas em sua operação para gestão e uso responsável de água?"
                AS gestao_agua,

            "A empresa possui práticas em sua operação para gestão e eficiência energética?"
                AS eficiencia_energetica,

            "A empresa possui práticas em sua operação para gerenciamento de resíduos?"
                AS gerenciamento_residuos,

            "A empresa possui práticas implementadas para calcular ou estimar a pegada de carbono de seus produtos e serviços?"
                AS calcula_pegada_carbono,

            "Caso tenha inventário de GEE, o mesmo é auditado por empresa terceira parte?"
                AS inventario_auditado,

            "A empresa possui práticas estruturadas para promover a inclusão e a diversidade entre seus colaboradores (ex.: equidade de gênero, raça, pessoas com deficiência, LGBTQIA+, etc)?"
                AS possui_diversidade,

            "A empresa realiza ações ou programas de voluntariado com envolvimento da comunidade local?"
                AS possui_voluntariado,

            "A empresa possui programas ou práticas estruturadas voltadas à saúde mental e bem-estar emocional dos colaboradores (ex.: apoio psicológico, campanhas internas, parcerias com profissionais ou plat..."
                AS possui_saude_mental,

            "Sua empresa possui uma política formal de Compras Sustentáveis, separada do Código de Conduta para Fornecedores, que inclua objetivos qualitativos e metas quantitativas para questões ambientais e ..."
                AS compras_sustentaveis,

            "A empresa possui algum critério socioambiental estabelecido para selecionar sua cadeia de fornecedores?"
                AS criterio_fornecedores,

            "Os contratos firmados com fornecedores incluem cláusulas específicas sobre requisitos ambientais, sociais, trabalhistas e de direitos humanos?"
                AS clausulas_esg_fornecedores,

            "Os profissionais de compras da sua empresa recebem treinamentos regulares sobre temas de sustentabilidade aplicados à cadeia de suprimentos?"
                AS treinamento_fornecedores,

            "A empresa possui um programa ou práticas estruturadas para promover a diversidade na sua cadeia de fornecedores?"
                AS diversidade_fornecedores,

            "Sua empresa realiza auditorias (presenciais ou remotas) nos fornecedores para verificar o cumprimento de requisitos de sustentabilidade?"
                AS auditoria_fornecedores,

            "Confirma que todas as informações fornecidas estão corretas?"
                AS dados_confirmados

        FROM "ESGreportingBRONZE"
        """

        df = pd.read_sql(query, engine)
        df.to_csv(filepath, index=False)
        df.to_sql(
            name="ESGreportingSILVER",
            con=engine,
            if_exists='replace',
            index=False
        )
        logging.info("Criação de tabela SILVER concluída.")
        
    except Exception as e:
        logging.error(e)