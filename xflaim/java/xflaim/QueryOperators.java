//------------------------------------------------------------------------------
// Desc:	QueryOperators
//
// Tabs:	3
//
//		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
// $Id$
//------------------------------------------------------------------------------

package xflaim;

/**
 * Provides list of valid query operators.
 */

public final class QueryOperators
{
	public static final int XFLM_UNKNOWN_OP					= 0;
	public static final int XFLM_AND_OP							= 1;
	public static final int XFLM_OR_OP							= 2;
	public static final int XFLM_NOT_OP							= 3;
	public static final int XFLM_EQ_OP							= 4;
	public static final int XFLM_NE_OP							= 5;
	public static final int XFLM_APPROX_EQ_OP					= 6;
	public static final int XFLM_LT_OP							= 7;
	public static final int XFLM_LE_OP							= 8;
	public static final int XFLM_GT_OP							= 9;
	public static final int XFLM_GE_OP							= 10;
	public static final int XFLM_BITAND_OP						= 11;
	public static final int XFLM_BITOR_OP						= 12;
	public static final int XFLM_BITXOR_OP						= 13;
	public static final int XFLM_MULT_OP						= 14;
	public static final int XFLM_DIV_OP							= 15;
	public static final int XFLM_MOD_OP							= 16;
	public static final int XFLM_PLUS_OP						= 17;
	public static final int XFLM_MINUS_OP						= 18;
	public static final int XFLM_NEG_OP							= 19;
	public static final int XFLM_LPAREN_OP						= 20;
	public static final int XFLM_RPAREN_OP						= 21;
	public static final int XFLM_COMMA_OP						= 22;
	public static final int XFLM_LBRACKET_OP					= 23;
	public static final int XFLM_RBRACKET_OP					= 24;
}

