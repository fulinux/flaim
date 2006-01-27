//-------------------------------------------------------------------------
// Desc:	XML Wrapper - class definitions
// Tabs:	3
//
//		Copyright (c) 2000,2002-2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: fxml.h 12329 2006-01-20 17:49:30 -0700 (Fri, 20 Jan 2006) ahodgkinson $
//-------------------------------------------------------------------------

#ifndef FXML_H
#define FXML_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// Constants

#define F_XML_DOCUMENT_TAG					40000
#define F_XML_XMLDECL_TAG					40001
#define F_XML_VERSION_TAG					40002
#define F_XML_ENCODING_TAG					40003
#define F_XML_SDDECL_TAG					40004
#define F_XML_COMMENT_TAG					40005
#define F_XML_NOTATIONDECL_TAG			40006
#define F_XML_SYSLITERAL_TAG				40007
#define F_XML_PUBIDLITERAL_TAG			40008
#define F_XML_EXTERNALID_TAG				40009
#define F_XML_ENTITYDECL_TAG				40010
#define F_XML_NDATADECL_TAG				40011
#define F_XML_ATTRDEF_TAG					40012
#define F_XML_ENTITYREF_TAG				40013
#define F_XML_CHARREF_TAG					40014
#define F_XML_PEREF_TAG						40015
#define F_XML_ATTNAME_TAG					40016
#define F_XML_ATTVAL_TAG					40017
#define F_XML_ELEMENTNAME_TAG				40018
#define F_XML_DOCTYPEDECL_TAG				40019
#define F_XML_ELEMENTVAL_TAG				40020
#define F_XML_CDATA_TAG						40021
#define F_XML_ELEMENTDECL_TAG				40022
#define F_XML_ATTLISTDECL_TAG				40023
#define F_XML_NAME_TAG						40024
#define F_XML_OCCURS_TAG					40025
#define F_XML_CHOICE_TAG					40026
#define F_XML_SEQ_TAG						40027
#define F_XML_MIXED_TAG						40028
#define F_XML_CONTENTSPEC_TAG				40029
#define F_XML_ATTTYPE_TAG					40030

#define F_XML_WHITESPACE_TAG				41000
#define F_XML_UNKNOWN_TAG					41001

// Characters

#define F_XML_UNI_LINEFEED			((FLMUNICODE)10)
#define F_XML_UNI_SPACE				((FLMUNICODE)32)
#define F_XML_UNI_BANG				((FLMUNICODE)33)
#define F_XML_UNI_QUOTE				((FLMUNICODE)34)
#define F_XML_UNI_POUND				((FLMUNICODE)35)
#define F_XML_UNI_DOLLAR			((FLMUNICODE)36)
#define F_XML_UNI_PERCENT			((FLMUNICODE)37)
#define F_XML_UNI_AMP				((FLMUNICODE)38)
#define F_XML_UNI_APOS				((FLMUNICODE)39)
#define F_XML_UNI_LPAREN			((FLMUNICODE)40)
#define F_XML_UNI_RPAREN			((FLMUNICODE)41)
#define F_XML_UNI_ASTERISK			((FLMUNICODE)42)
#define F_XML_UNI_PLUS				((FLMUNICODE)43)
#define F_XML_UNI_COMMA				((FLMUNICODE)44)
#define F_XML_UNI_HYPHEN			((FLMUNICODE)45)
#define F_XML_UNI_PERIOD			((FLMUNICODE)46)
#define F_XML_UNI_FSLASH			((FLMUNICODE)47)

#define F_XML_UNI_0					((FLMUNICODE)48)
#define F_XML_UNI_1					((FLMUNICODE)49)
#define F_XML_UNI_2					((FLMUNICODE)50)
#define F_XML_UNI_3					((FLMUNICODE)51)
#define F_XML_UNI_4					((FLMUNICODE)52)
#define F_XML_UNI_5					((FLMUNICODE)53)
#define F_XML_UNI_6					((FLMUNICODE)54)
#define F_XML_UNI_7					((FLMUNICODE)55)
#define F_XML_UNI_8					((FLMUNICODE)56)
#define F_XML_UNI_9					((FLMUNICODE)57)

#define F_XML_UNI_COLON				((FLMUNICODE)58)
#define F_XML_UNI_SEMI				((FLMUNICODE)59)
#define F_XML_UNI_LT					((FLMUNICODE)60)
#define F_XML_UNI_EQ					((FLMUNICODE)61)
#define F_XML_UNI_GT					((FLMUNICODE)62)
#define F_XML_UNI_QUEST				((FLMUNICODE)63)
#define F_XML_UNI_ATSIGN			((FLMUNICODE)64)

#define F_XML_UNI_A					((FLMUNICODE)65)
#define F_XML_UNI_B					((FLMUNICODE)66)
#define F_XML_UNI_C					((FLMUNICODE)67)
#define F_XML_UNI_D					((FLMUNICODE)68)
#define F_XML_UNI_E					((FLMUNICODE)69)
#define F_XML_UNI_F					((FLMUNICODE)70)
#define F_XML_UNI_G					((FLMUNICODE)71)
#define F_XML_UNI_H					((FLMUNICODE)72)
#define F_XML_UNI_I					((FLMUNICODE)73)
#define F_XML_UNI_J					((FLMUNICODE)74)
#define F_XML_UNI_K					((FLMUNICODE)75)
#define F_XML_UNI_L					((FLMUNICODE)76)
#define F_XML_UNI_M					((FLMUNICODE)77)
#define F_XML_UNI_N					((FLMUNICODE)78)
#define F_XML_UNI_O					((FLMUNICODE)79)
#define F_XML_UNI_P					((FLMUNICODE)80)
#define F_XML_UNI_Q					((FLMUNICODE)81)
#define F_XML_UNI_R					((FLMUNICODE)82)
#define F_XML_UNI_S					((FLMUNICODE)83)
#define F_XML_UNI_T					((FLMUNICODE)84)
#define F_XML_UNI_U					((FLMUNICODE)85)
#define F_XML_UNI_V					((FLMUNICODE)86)
#define F_XML_UNI_W					((FLMUNICODE)87)
#define F_XML_UNI_X					((FLMUNICODE)88)
#define F_XML_UNI_Y					((FLMUNICODE)89)
#define F_XML_UNI_Z					((FLMUNICODE)90)

#define F_XML_UNI_LBRACKET			((FLMUNICODE)91)
#define F_XML_UNI_RBRACKET			((FLMUNICODE)93)
#define F_XML_UNI_UNDERSCORE		((FLMUNICODE)95)

#define F_XML_UNI_a					((FLMUNICODE)97)
#define F_XML_UNI_b					((FLMUNICODE)98)
#define F_XML_UNI_c					((FLMUNICODE)99)
#define F_XML_UNI_d					((FLMUNICODE)100)
#define F_XML_UNI_e					((FLMUNICODE)101)
#define F_XML_UNI_f					((FLMUNICODE)102)
#define F_XML_UNI_g					((FLMUNICODE)103)
#define F_XML_UNI_h					((FLMUNICODE)104)
#define F_XML_UNI_i					((FLMUNICODE)105)
#define F_XML_UNI_j					((FLMUNICODE)106)
#define F_XML_UNI_k					((FLMUNICODE)107)
#define F_XML_UNI_l					((FLMUNICODE)108)
#define F_XML_UNI_m					((FLMUNICODE)109)
#define F_XML_UNI_n					((FLMUNICODE)110)
#define F_XML_UNI_o					((FLMUNICODE)111)
#define F_XML_UNI_p					((FLMUNICODE)112)
#define F_XML_UNI_q					((FLMUNICODE)113)
#define F_XML_UNI_r					((FLMUNICODE)114)
#define F_XML_UNI_s					((FLMUNICODE)115)
#define F_XML_UNI_t					((FLMUNICODE)116)
#define F_XML_UNI_u					((FLMUNICODE)117)
#define F_XML_UNI_v					((FLMUNICODE)118)
#define F_XML_UNI_w					((FLMUNICODE)119)
#define F_XML_UNI_x					((FLMUNICODE)120)
#define F_XML_UNI_y					((FLMUNICODE)121)
#define F_XML_UNI_z					((FLMUNICODE)122)

#define F_XML_UNI_PIPE				((FLMUNICODE)124)
#define F_XML_UNI_TILDE				((FLMUNICODE)126)

// Typedefs

typedef struct xmlChar
{
	FLMBYTE		ucFlags;
} XMLCHAR;

class F_XML : public F_Base
{
public:

	F_XML();

	virtual ~F_XML();
	
	FLMBOOL isPubidChar(
		FLMUNICODE		uChar);

	FLMBOOL isQuoteChar(
		FLMUNICODE		uChar);

	FLMBOOL isWhitespace(
		FLMUNICODE		uChar);

	FLMBOOL isExtender(
		FLMUNICODE		uChar);

	FLMBOOL isCombiningChar(
		FLMUNICODE		uChar);

	FLMBOOL isNameChar(
		FLMUNICODE		uChar);

	FLMBOOL isIdeographic(
		FLMUNICODE		uChar);

	FLMBOOL isBaseChar(
		FLMUNICODE		uChar);

	FLMBOOL isDigit(
		FLMUNICODE		uChar);

	FLMBOOL isLetter(
		FLMUNICODE		uChar);

	void setCharFlag(
		FLMUNICODE		uLowChar,
		FLMUNICODE		uHighChar,
		FLMUINT16		ui16Flag);

	FLMBOOL isNameValid(
		const FLMUNICODE *	puzName,
		const char *			pszName);

	RCODE buildCharTable( void);

protected:

	POOL					m_tmpPool;

private:

	XMLCHAR *			m_pCharTable;
};

/****************************************************************************
Desc: 	FLAIM's XML export class
****************************************************************************/
class F_XMLExport : public F_XML
{
public:

	F_XMLExport();
	
	virtual ~F_XMLExport();

	RCODE setup( void);

	// Methods

	RCODE exportRecord(
		F_NameTable *	pNameTable,
		FlmRecord *		pRec,
		FLMUINT			uiStartIndent,
		FLMUINT			uiIndentSize,
		POOL *			pPool,
		char **			ppszXML,
		FLMUINT *		puiBytes);

private:

	FLMBYTE			m_szSpaces[ 256];
	FLMUINT			m_uiTmpBufSize;
	FLMBYTE *		m_pszTmpBuf;
	FCS_BIOS *		m_pByteStream;
	FLMBOOL			m_bSetup;
};

typedef enum eFlmXMLTokenType
{
	FLM_XML_STAG,
	FLM_XML_ETAG,
	FLM_XML_COMMENT,
	FLM_XML_CHAR_DATA
} FlmXMLTokenType;

/****************************************************************************
Desc: 	FLAIM's XML import class
****************************************************************************/
class F_XMLImport: public F_XML
{
public:

	F_XMLImport();
	
	virtual ~F_XMLImport();

	RCODE setup( void);

	RCODE importDocument(
		HFDB				hDb,
		F_NameTable *	pNameTable,
		FCS_ISTM *		pStream,
		FLMBOOL			bSubset,
		FlmRecord **	ppRecord);

	void reset( void);

private:

	RCODE getFieldTagAndType(
		FLMUNICODE *	puzName,
		FLMUINT *		puiTagNum,
		FLMUINT *		puiDataType);

	RCODE getByte(
		FLMBYTE *		pucByte);

	RCODE ungetByte(
		FLMBYTE 			ucByte);

	RCODE ungetChar(
		FLMUNICODE 		uChar);

	RCODE ungetChars(
		FLMUNICODE *	puChars,
		FLMUINT			uiChars);

	RCODE peekChar(
		FLMUNICODE *	puChar);

	RCODE getName(
		FLMUNICODE *	puzName,
		FLMUINT *		puiChars);

	RCODE getNmtoken(
		FLMUNICODE *	puzName,
		FLMUINT *		puiChars);

	RCODE getPubidLiteral(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiMaxChars);

	RCODE getSystemLiteral(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiMaxChars);

	RCODE	getChar(
		FLMUNICODE *	pChar);

	RCODE getChars(
		FLMUNICODE *	uzChars,
		FLMUINT *		puiCount);

	RCODE getCharEntity(
		FLMUNICODE *	puChar);

	RCODE getElementValue(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiMaxChars,
		FLMBOOL *		pbEntity);

	RCODE processEntityValue(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE getEntity(
		FLMUNICODE *	puBuf,
		FLMUINT *		puiChars,
		FLMBOOL *		pbTranslated,
		FLMUNICODE *	puTransChar);

	RCODE processReference(
		FlmRecord *		pRec,
		void *			pvParent,
		FLMUNICODE *	puChar = NULL);

	RCODE processCDATA(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processAttributes(
		HFDB				hDb,
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processComment(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processProlog(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processXMLDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processVersion(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processEncodingDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processSDDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processMisc(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processDocTypeDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processPI(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processElement(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE unicodeToNumber(
		const FLMUNICODE *	puzVal,
		FLMUINT *				puiVal,
		FLMBOOL *				pbNeg);

	RCODE setElementValue(
		FlmRecord *				pRec,
		void *					pvField,
		const FLMUNICODE *	puzValue);

	RCODE processAttributes(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processMarkupDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processPERef(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processElementDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processEntityDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processNotationDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processAttListDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processContentSpec(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processMixedContent(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processChildContent(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processAttDef(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processAttType(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processAttValue(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processDefaultDecl(
		FlmRecord *		pRec,
		void *			pvParent);

	RCODE processID(
		FlmRecord *		pRec,
		void *			pvParent,
		FLMBOOL *		pbPublicId);

	RCODE processSTag(
		FlmRecord *		pRec,
		void *			pvParent,
		FLMBOOL *		pbHasContent,
		void **			ppvElementRoot);

	RCODE skipWhitespace(
		void *			pvParent,
		FLMBOOL 			bRequired = FALSE);

	RCODE skipName( void);

	RCODE skipEntity( void);

	RCODE isXMLDecl(
		FLMBOOL *		pbIsXMLDecl);

	RCODE isDocTypeDecl(
		FLMBOOL *		pbIsDocTypeDecl);

	// Data

	FLMBOOL				m_bSubset;
	FLMUINT				m_uiUngetPos;
#define F_XML_MAX_UNGET			32
	FLMUNICODE			m_puUngetBuf[ F_XML_MAX_UNGET];
#define F_XML_MAX_CHARS			128
	FLMUNICODE			m_uChars[ F_XML_MAX_CHARS];
	FLMBYTE				m_ucUngetByte;
	FLMBOOL				m_bSetup;
	FCS_ISTM *			m_pStream;
	FLMUNICODE *		m_puValBuf;
	FLMUINT				m_uiValBufSize; // Number of Unicode characters
	F_NameTable *		m_pNameTable;
	HFDB					m_hDb;
};

#include "fpackoff.h"

#endif // FXML_H
